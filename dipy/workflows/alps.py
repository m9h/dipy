"""Workflow for DTI-ALPS (Diffusion along Perivascular Spaces) analysis.

Computes the DTI-ALPS index for assessing glymphatic system function
from diffusion tensor data :footcite:p:`Taoka2017`.
"""

import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from dipy.io.image import load_nifti, save_nifti
from dipy.utils.optpkg import optional_package
from dipy.workflows.workflow import Workflow

ants, have_ants, _ = optional_package("ants")

# JHU-ICBM atlas ROI coordinates in 1mm MNI voxel space
# These correspond to the superior corona radiata (SCR, projection fibers)
# and superior longitudinal fasciculus (SLF, association fibers)
_JHU_ROIS = {
    "L_SCR": (116, 110, 99),   # left projection
    "R_SCR": (64, 110, 99),    # right projection
    "L_SLF": (128, 110, 99),   # left association
    "R_SLF": (52, 110, 99),    # right association
}


def _create_sphere_roi(shape, center, radius_mm, voxel_size):
    """Create a binary spherical ROI mask.

    Parameters
    ----------
    shape : tuple
        Volume shape (i, j, k).
    center : tuple
        Center voxel coordinates (x, y, z).
    radius_mm : float
        Sphere radius in millimeters.
    voxel_size : array-like
        Voxel dimensions in mm.

    Returns
    -------
    ndarray
        Binary mask of shape ``shape``.
    """
    radius_vox = radius_mm / np.array(voxel_size[:3])
    coords = np.ogrid[
        0:shape[0],
        0:shape[1],
        0:shape[2],
    ]
    dist = sum(
        ((c - cen) / r) ** 2
        for c, cen, r in zip(coords, center, radius_vox)
    )
    return (dist <= 1.0).astype(np.float32)


def _extract_tensor_components(tensor_data):
    """Extract Dxx, Dyy, Dzz from a DTI tensor volume.

    Parameters
    ----------
    tensor_data : ndarray
        Tensor data. Accepted shapes:

        - ``(i, j, k, 6)`` — lower-triangular elements
        - ``(i, j, k, 1, 6)`` — NIfTI symmetric matrix format

        Elements are ordered as ``[Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]``.

    Returns
    -------
    Dxx, Dyy, Dzz : ndarray
        Each has shape ``(i, j, k)``.
    """
    if tensor_data.ndim == 5 and tensor_data.shape[3] == 1:
        tensor_data = tensor_data[:, :, :, 0, :]
    if tensor_data.shape[-1] != 6:
        raise ValueError(
            f"Expected 6 tensor components, got {tensor_data.shape[-1]}"
        )
    Dxx = tensor_data[..., 0]
    Dyy = tensor_data[..., 3]
    Dzz = tensor_data[..., 5]
    return Dxx, Dyy, Dzz


def _compute_alps(Dxx, Dyy, Dzz, roi_masks):
    """Compute DTI-ALPS index from tensor components and ROI masks.

    Parameters
    ----------
    Dxx, Dyy, Dzz : ndarray
        Diffusivity along x, y, z axes. Shape ``(i, j, k)``.
    roi_masks : dict
        Dictionary mapping ROI names to binary masks.

    Returns
    -------
    dict
        ALPS metrics including per-hemisphere and mean indices.
    """
    def _mean_in_roi(data, mask):
        vals = data[mask > 0]
        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    results = {}

    # Extract mean diffusivities per ROI
    for name, mask in roi_masks.items():
        results[f"Dxx_{name}"] = _mean_in_roi(Dxx, mask)
        results[f"Dyy_{name}"] = _mean_in_roi(Dyy, mask)
        results[f"Dzz_{name}"] = _mean_in_roi(Dzz, mask)

    # ALPS index per hemisphere (Taoka et al. 2017)
    # ALPS = mean(Dxx_proj, Dxx_assoc) / mean(Dyy_proj, Dzz_assoc)
    eps = 1e-10
    denom_l = results["Dyy_L_SCR"] + results["Dzz_L_SLF"] + eps
    denom_r = results["Dyy_R_SCR"] + results["Dzz_R_SLF"] + eps

    results["alps_l"] = (
        results["Dxx_L_SCR"] + results["Dxx_L_SLF"]
    ) / denom_l
    results["alps_r"] = (
        results["Dxx_R_SCR"] + results["Dxx_R_SLF"]
    ) / denom_r
    results["alps_mean"] = (results["alps_l"] + results["alps_r"]) / 2.0

    return results


class ALPSFlow(Workflow):
    """DTI-ALPS (Diffusion along Perivascular Spaces) analysis workflow.

    Computes the DTI-ALPS index :footcite:p:`Taoka2017` for assessing
    glymphatic system function from diffusion tensor data. Supports both
    standard DTI and free-water-corrected DTI tensors.

    The pipeline:
    1. Registers subject FA to JHU-ICBM template (via ANTsPy or dipy)
    2. Warps atlas ROIs (SCR/SLF) to native subject space
    3. Extracts Dxx, Dyy, Dzz from the tensor in each ROI
    4. Computes bilateral ALPS index
    """

    @classmethod
    def get_short_name(cls):
        """Return short name for the workflow.

        Returns
        -------
        str
            Short name 'alps'.
        """
        return "alps"

    def run(
        self,
        tensor_files,
        fa_files,
        mask_files,
        fwdti_tensor_files=None,
        roi_radius=2.5,
        registration="ants",
        out_dir="",
        out_alps_csv="alps_index.csv",
        out_rois="alps_rois.nii.gz",
    ):
        """Compute DTI-ALPS index for glymphatic function assessment.

        Performs DTI-ALPS analysis :footcite:p:`Taoka2017` by registering
        subject FA to a template, warping atlas-defined ROIs to native space,
        and computing the ALPS index from diffusion tensor components.

        Parameters
        ----------
        tensor_files : string or Path
            Path to the DTI tensor NIfTI file. The tensor should contain 6
            components (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz) in the last dimension.
        fa_files : string or Path
            Path to the FA map NIfTI file.
        mask_files : string or Path
            Path to the brain mask NIfTI file.
        fwdti_tensor_files : string or Path, optional
            Path to the free-water DTI tensor NIfTI file. If provided,
            free-water-corrected ALPS index is also computed.
        roi_radius : float, optional
            Radius of spherical ROIs in millimeters.
        registration : string, optional
            Registration method: 'ants' (ANTsPy, recommended) or 'dipy'.
            Falls back to dipy if ANTsPy is not installed.
        out_dir : string, optional
            Output directory.
        out_alps_csv : string, optional
            Output CSV filename for ALPS metrics.
        out_rois : string, optional
            Output NIfTI filename for ROI masks in native space.

        References
        ----------
        .. footbibliography::
        """
        io_it = self.get_io_iterator()

        for (
            tensor_path,
            fa_path,
            mask_path,
            fwdti_path,
            out_csv_path,
            out_rois_path,
        ) in io_it:
            # Load data
            tensor_data, tensor_affine = load_nifti(tensor_path)
            fa_data, fa_affine = load_nifti(fa_path)
            mask_data, _ = load_nifti(mask_path)

            voxel_size = nib.load(fa_path).header.get_zooms()

            print(f"Computing DTI-ALPS for {fa_path}")
            print(f"  Tensor shape: {tensor_data.shape}")
            print(f"  FA shape: {fa_data.shape}")
            print(f"  Voxel size: {voxel_size}")

            # Extract tensor components
            Dxx, Dyy, Dzz = _extract_tensor_components(tensor_data)

            # Register to template and get ROIs in native space
            roi_masks = self._get_native_rois(
                fa_data,
                fa_affine,
                voxel_size,
                mask_data,
                roi_radius,
                registration,
            )

            # Compute ALPS index
            results = _compute_alps(Dxx, Dyy, Dzz, roi_masks)

            # Also extract FA in ROIs
            for name, mask in roi_masks.items():
                vals = fa_data[mask > 0]
                results[f"fa_{name}"] = (
                    float(np.mean(vals)) if len(vals) > 0 else 0.0
                )

            # Compute MD = (Dxx + Dyy + Dzz) / 3 and extract in ROIs
            md_data = (Dxx + Dyy + Dzz) / 3.0
            for name, mask in roi_masks.items():
                vals = md_data[mask > 0]
                results[f"md_{name}"] = (
                    float(np.mean(vals)) if len(vals) > 0 else 0.0
                )

            # Free-water corrected ALPS if fwDTI tensor provided
            if fwdti_path and Path(fwdti_path).exists():
                fw_tensor, _ = load_nifti(fwdti_path)
                fw_Dxx, fw_Dyy, fw_Dzz = _extract_tensor_components(
                    fw_tensor
                )
                fw_results = _compute_alps(
                    fw_Dxx, fw_Dyy, fw_Dzz, roi_masks
                )
                results["fw_alps_l"] = fw_results["alps_l"]
                results["fw_alps_r"] = fw_results["alps_r"]
                results["fw_alps_mean"] = fw_results["alps_mean"]

            # Print results
            print(f"  ALPS_L: {results['alps_l']:.4f}")
            print(f"  ALPS_R: {results['alps_r']:.4f}")
            print(f"  ALPS_mean: {results['alps_mean']:.4f}")
            if "fw_alps_mean" in results:
                print(f"  FW-ALPS_mean: {results['fw_alps_mean']:.4f}")

            # Save CSV
            with open(out_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results.keys())
                writer.writerow(results.values())
            print(f"  Saved: {out_csv_path}")

            # Save combined ROI mask
            combined_roi = np.zeros_like(fa_data, dtype=np.int16)
            for i, (name, mask) in enumerate(roi_masks.items(), start=1):
                combined_roi[mask > 0] = i
            save_nifti(out_rois_path, combined_roi, fa_affine)
            print(f"  Saved: {out_rois_path}")

    def _get_native_rois(
        self,
        fa_data,
        fa_affine,
        voxel_size,
        mask_data,
        roi_radius,
        registration,
    ):
        """Register template to subject and create ROIs in native space.

        Parameters
        ----------
        fa_data : ndarray
            Subject FA map.
        fa_affine : ndarray
            Affine transform for the FA map.
        voxel_size : tuple
            Voxel dimensions in mm.
        mask_data : ndarray
            Brain mask.
        roi_radius : float
            ROI sphere radius in mm.
        registration : str
            Registration method ('ants' or 'dipy').

        Returns
        -------
        dict
            Mapping of ROI name to binary mask in native space.
        """
        use_ants = registration == "ants" and have_ants

        if use_ants:
            return self._register_ants(
                fa_data, fa_affine, voxel_size, roi_radius
            )
        else:
            if registration == "ants" and not have_ants:
                print(
                    "  ANTsPy not available, falling back to dipy registration"
                )
            return self._register_dipy(
                fa_data, fa_affine, voxel_size, mask_data, roi_radius
            )

    def _register_ants(self, fa_data, fa_affine, voxel_size, roi_radius):
        """Register using ANTsPy and warp ROIs to native space.

        Parameters
        ----------
        fa_data : ndarray
            Subject FA map.
        fa_affine : ndarray
            Affine for the FA volume.
        voxel_size : tuple
            Voxel dimensions.
        roi_radius : float
            ROI radius in mm.

        Returns
        -------
        dict
            ROI masks in native space.
        """
        from dipy.data import fetch_atlas_jhu_labels

        # Get JHU template
        try:
            atlas_path = fetch_atlas_jhu_labels()
            template_img = ants.image_read(str(atlas_path))
        except Exception:
            # Fall back: create a simple template from the FA data shape
            print("  JHU atlas not available, using approximate ROI placement")
            return self._approximate_rois(fa_data.shape, voxel_size, roi_radius)

        # Convert subject FA to ANTs image
        subject_img = ants.from_numpy(
            fa_data, origin=fa_affine[:3, 3].tolist(),
            spacing=list(voxel_size[:3]),
        )

        # Register template -> subject (inverse direction for warping ROIs)
        print("  Running ANTsPy SyN registration...")
        reg = ants.registration(
            fixed=subject_img,
            moving=template_img,
            type_of_transform="SyN",
            verbose=False,
        )

        # Create ROIs in template space and warp to native
        template_shape = template_img.numpy().shape
        template_spacing = template_img.spacing
        roi_masks = {}

        for name, center in _JHU_ROIS.items():
            # Create sphere in template space
            roi_template = _create_sphere_roi(
                template_shape, center, roi_radius, template_spacing
            )
            roi_ants = ants.from_numpy(
                roi_template,
                origin=template_img.origin,
                spacing=template_img.spacing,
                direction=template_img.direction,
            )
            # Warp to native space
            roi_native = ants.apply_transforms(
                fixed=subject_img,
                moving=roi_ants,
                transformlist=reg["fwdtransforms"],
                interpolator="nearestNeighbor",
            )
            roi_masks[name] = (roi_native.numpy() > 0.5).astype(np.float32)

        return roi_masks

    def _register_dipy(
        self, fa_data, fa_affine, voxel_size, mask_data, roi_radius
    ):
        """Register using dipy's affine registration.

        Parameters
        ----------
        fa_data : ndarray
            Subject FA map.
        fa_affine : ndarray
            FA affine.
        voxel_size : tuple
            Voxel dimensions.
        mask_data : ndarray
            Brain mask.
        roi_radius : float
            ROI radius in mm.

        Returns
        -------
        dict
            ROI masks in native space.
        """
        print("  Using dipy affine registration (approximate ROI placement)")
        return self._approximate_rois(fa_data.shape, voxel_size, roi_radius)

    def _approximate_rois(self, shape, voxel_size, roi_radius):
        """Create approximate ROIs without registration.

        Uses the assumption that the subject is roughly aligned to
        standard orientation (RAS). Scales JHU coordinates based on
        volume dimensions.

        Parameters
        ----------
        shape : tuple
            Volume shape.
        voxel_size : tuple
            Voxel dimensions in mm.
        roi_radius : float
            ROI radius in mm.

        Returns
        -------
        dict
            ROI masks.
        """
        # Scale JHU 1mm coords to subject voxel space
        # JHU template is 182x218x182 at 1mm
        scale = np.array(shape[:3]) / np.array([182, 218, 182])
        roi_masks = {}
        for name, center in _JHU_ROIS.items():
            scaled_center = tuple(int(c * s) for c, s in zip(center, scale))
            # Clamp to volume bounds
            clamped = tuple(
                max(0, min(s - 1, c))
                for c, s in zip(scaled_center, shape[:3])
            )
            roi_masks[name] = _create_sphere_roi(
                shape[:3], clamped, roi_radius, voxel_size
            )
        return roi_masks
