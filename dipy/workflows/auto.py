"""Automated diffusion MRI processing pipeline workflow.

This module provides ``AutoPipelineFlow``, a DIPY workflow that runs a
full or partial diffusion MRI processing pipeline. It supports two
modes of operation:

1. **BIDS mode**: Pass a BIDS dataset directory and output directory as
   positional arguments. The pipeline discovers subjects and runs
   processing per-subject, writing outputs in BIDS derivatives format.

2. **File mode** (legacy): Pass individual input files directly, as
   with other DIPY CLI workflows.

The pipeline stages include denoising, brain masking, DTI fitting, and
optionally tracking.
"""

import json
from pathlib import Path
import sys

from dipy.io.bids import is_bids_dir, load_bids_layout, write_derivative_description
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.utils.logging import logger
from dipy.workflows.workflow import Workflow


class AutoPipelineFlow(Workflow):
    """Automated diffusion MRI processing pipeline.

    Supports both BIDS app mode and legacy file-based mode. In BIDS
    mode, the first positional argument is a BIDS directory (containing
    ``dataset_description.json``), and outputs are written in BIDS
    derivative format.
    """

    @classmethod
    def get_short_name(cls):
        return "auto"

    def run(
        self,
        input_path,
        output_path="",
        *,
        analysis_level="participant",
        participant_label=None,
        session_label=None,
        pipeline_type="full",
        b0_threshold=50,
        denoise_method="patch2self",
        brain_mask_method="synthseg",
        dti_fit=True,
        tracking=False,
        out_dir="",
        out_pipeline_log="pipeline_log.json",
    ):
        """Run the automated diffusion MRI processing pipeline.

        This workflow supports two modes of operation:

        **BIDS mode**: When ``input_path`` is a BIDS dataset directory
        (containing ``dataset_description.json``), the pipeline discovers
        subjects and sessions, processes each subject's DWI data, and
        writes outputs in BIDS derivative format under ``output_path``.

        **File mode**: When ``input_path`` is a file path (or glob
        pattern), it operates like a standard DIPY workflow. Use
        ``--out_dir`` to control output location.

        Parameters
        ----------
        input_path : string or Path
            Path to a BIDS dataset directory or to input DWI file(s).
            In BIDS mode, this must be a directory containing
            ``dataset_description.json``. In file mode, this path may
            contain wildcards to process multiple inputs at once.
        output_path : string or Path, optional
            Output directory for BIDS derivative results. Used only in
            BIDS mode. In file mode, use ``--out_dir`` instead.
        analysis_level : string, optional
            BIDS analysis level. Currently only ``'participant'`` is
            supported.
        participant_label : variable string, optional
            One or more participant labels to process (e.g.,
            ``sub-01 sub-02``). If not specified, all participants in the
            BIDS dataset are processed. Only used in BIDS mode.
        session_label : variable string, optional
            One or more session labels to process (e.g., ``ses-01``).
            If not specified, all sessions are processed. Only used in
            BIDS mode.
        pipeline_type : string, optional
            Type of processing pipeline to run. Options are:

            - ``'full'``: Run all stages (denoise, mask, DTI fit).
            - ``'denoise'``: Run only denoising.
            - ``'mask'``: Run only brain masking.
            - ``'dti'``: Run only DTI fitting.

        b0_threshold : int, optional
            Threshold to consider a volume as b0.
        denoise_method : string, optional
            Denoising method to use. Options are ``'patch2self'``,
            ``'nlmeans'``, ``'mppca'``, ``'lpca'``.
        brain_mask_method : string, optional
            Brain masking method. Options are ``'synthseg'``,
            ``'median_otsu'``, ``'evac'``.
        dti_fit : bool, optional
            Whether to run DTI model fitting.
        tracking : bool, optional
            Whether to run fiber tracking after model fitting.
        out_dir : string or Path, optional
            Output directory. Used in file mode. In BIDS mode,
            ``output_path`` is used instead.
        out_pipeline_log : string, optional
            Name of the pipeline log file to be saved.
        """
        valid_pipelines = ["full", "denoise", "mask", "dti"]
        if pipeline_type not in valid_pipelines:
            logger.error(
                f"Unknown pipeline_type '{pipeline_type}'. "
                f"Choose one of: {', '.join(valid_pipelines)}."
            )
            sys.exit(1)

        if analysis_level != "participant":
            logger.error(
                f"Analysis level '{analysis_level}' is not supported. "
                "Only 'participant' level is currently implemented."
            )
            sys.exit(1)

        # Detect BIDS mode vs file mode
        if is_bids_dir(input_path):
            _output_dir = (
                Path(output_path) if output_path
                else Path(input_path) / "derivatives" / "dipy"
            )
            return self._run_bids(
                bids_dir=Path(input_path),
                output_dir=_output_dir,
                participant_label=participant_label,
                session_label=session_label,
                pipeline_type=pipeline_type,
                b0_threshold=b0_threshold,
                denoise_method=denoise_method,
                brain_mask_method=brain_mask_method,
                dti_fit=dti_fit,
                tracking=tracking,
            )

        # ---- File mode ----
        # get_io_iterator() inspects this frame, so it must be called
        # directly inside run() where the local variable names match the
        # run() parameter names.
        logger.info("Running DIPY auto pipeline in file mode")
        logger.info(f"  Input: {input_path}")

        io_it = self.get_io_iterator()

        for fpath, log_out_path in io_it:
            logger.info(f"Processing {fpath}")

            # Determine output directory from the log output path
            file_out_dir = Path(log_out_path).parent
            file_stem = Path(fpath).name.split(".")[0]

            result = {"status": "completed", "outputs": {}}

            # Look for bval/bvec files alongside the input
            bval_path = None
            bvec_path = None
            input_p = Path(fpath)
            stem = input_p.name
            for ext in [".nii.gz", ".nii"]:
                if stem.endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            candidate_bval = input_p.parent / (stem + ".bval")
            candidate_bvec = input_p.parent / (stem + ".bvec")
            if candidate_bval.is_file():
                bval_path = str(candidate_bval)
            if candidate_bvec.is_file():
                bvec_path = str(candidate_bvec)

            current_dwi = str(fpath)

            # Stage 1: Denoising
            if pipeline_type in ("full", "denoise"):
                denoised_path = str(
                    file_out_dir / f"{file_stem}_denoised.nii.gz"
                )
                try:
                    self._run_denoise(
                        dwi_path=current_dwi,
                        bval_path=bval_path,
                        output_path=denoised_path,
                        method=denoise_method,
                        b0_threshold=b0_threshold,
                    )
                    current_dwi = denoised_path
                    result["outputs"]["denoised"] = denoised_path
                except Exception as e:
                    logger.error(f"Denoising failed: {e}")

            # Stage 2: Brain masking
            if pipeline_type in ("full", "mask"):
                mask_path = str(
                    file_out_dir / f"{file_stem}_brain_mask.nii.gz"
                )
                try:
                    self._run_brain_mask(
                        input_path=current_dwi,
                        output_path=mask_path,
                        method=brain_mask_method,
                        bval_path=bval_path,
                        b0_threshold=b0_threshold,
                    )
                    result["outputs"]["brain_mask"] = mask_path
                except Exception as e:
                    logger.error(f"Brain masking failed: {e}")
                    mask_path = None

            # Stage 3: DTI fitting
            if pipeline_type in ("full", "dti") and dti_fit:
                if bval_path and bvec_path:
                    fa_path = str(file_out_dir / f"{file_stem}_fa.nii.gz")
                    md_path = str(file_out_dir / f"{file_stem}_md.nii.gz")
                    rd_path = str(file_out_dir / f"{file_stem}_rd.nii.gz")
                    ad_path = str(file_out_dir / f"{file_stem}_ad.nii.gz")
                    rgb_path = str(
                        file_out_dir / f"{file_stem}_color_fa.nii.gz"
                    )
                    try:
                        self._run_dti(
                            dwi_path=current_dwi,
                            bval_path=bval_path,
                            bvec_path=bvec_path,
                            mask_path=result["outputs"].get("brain_mask"),
                            fa_path=fa_path,
                            md_path=md_path,
                            rd_path=rd_path,
                            ad_path=ad_path,
                            rgb_path=rgb_path,
                            b0_threshold=b0_threshold,
                        )
                        result["outputs"]["FA"] = fa_path
                        result["outputs"]["MD"] = md_path
                    except Exception as e:
                        logger.error(f"DTI fitting failed: {e}")
                else:
                    logger.warning(
                        "No bval/bvec files found. Skipping DTI fitting."
                    )

            # Write pipeline log
            log_data = {
                "input": str(fpath),
                "pipeline_type": pipeline_type,
                "outputs": result["outputs"],
            }
            with open(log_out_path, "w") as f:
                json.dump(log_data, f, indent=2)
                f.write("\n")
            logger.info(f"Pipeline log saved to {log_out_path}")

        logger.info("File-mode auto pipeline completed.")
        return io_it

    # ------------------------------------------------------------------
    #  BIDS mode
    # ------------------------------------------------------------------

    def _run_bids(
        self,
        bids_dir,
        output_dir,
        participant_label,
        session_label,
        pipeline_type,
        b0_threshold,
        denoise_method,
        brain_mask_method,
        dti_fit,
        tracking,
    ):
        """Run the pipeline in BIDS mode.

        Parameters
        ----------
        bids_dir : Path
            Root of the BIDS dataset.
        output_dir : Path
            Derivatives output directory.
        participant_label : list of str or None
            Participant labels to process.
        session_label : list of str or None
            Session labels to process.
        pipeline_type : str
            Pipeline type.
        b0_threshold : int
            B0 threshold.
        denoise_method : str
            Denoising method.
        brain_mask_method : str
            Brain masking method.
        dti_fit : bool
            Whether to fit DTI.
        tracking : bool
            Whether to run tracking.

        Returns
        -------
        dict
            Results dictionary keyed by subject/session.
        """
        logger.info("Running DIPY auto pipeline in BIDS mode")
        logger.info(f"  BIDS directory: {bids_dir}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Pipeline type: {pipeline_type}")

        # Parse BIDS layout
        layout = load_bids_layout(
            bids_dir,
            participant_label=participant_label,
            session_label=session_label,
        )

        if not layout["subjects"]:
            logger.error("No subjects found in the BIDS dataset.")
            sys.exit(1)

        # Write derivative dataset_description.json
        source_desc = layout["dataset_description"].copy()
        source_desc["bids_dir"] = str(bids_dir)
        write_derivative_description(
            output_dir,
            source_dataset=source_desc,
        )

        # Process each subject/session
        results = {}
        for sub_label, sessions in layout["subjects"].items():
            for ses_label, modalities in sessions.items():
                logger.info(f"Processing {sub_label}/{ses_label}")
                result = self._process_subject_session(
                    sub_label=sub_label,
                    ses_label=ses_label,
                    modalities=modalities,
                    output_dir=output_dir,
                    pipeline_type=pipeline_type,
                    b0_threshold=b0_threshold,
                    denoise_method=denoise_method,
                    brain_mask_method=brain_mask_method,
                    dti_fit=dti_fit,
                    tracking=tracking,
                )
                results[f"{sub_label}/{ses_label}"] = result

        # Write pipeline log
        log_path = output_dir / "pipeline_log.json"
        pipeline_log = {
            "pipeline_type": pipeline_type,
            "denoise_method": denoise_method,
            "brain_mask_method": brain_mask_method,
            "dti_fit": dti_fit,
            "tracking": tracking,
            "b0_threshold": b0_threshold,
            "subjects_processed": list(results.keys()),
        }
        with open(log_path, "w") as f:
            json.dump(pipeline_log, f, indent=2)
            f.write("\n")

        logger.info(f"Pipeline log saved to {log_path}")
        logger.info("BIDS auto pipeline completed.")
        return results

    def _process_subject_session(
        self,
        sub_label,
        ses_label,
        modalities,
        output_dir,
        pipeline_type,
        b0_threshold,
        denoise_method,
        brain_mask_method,
        dti_fit,
        tracking,
    ):
        """Process a single subject/session through the pipeline.

        Parameters
        ----------
        sub_label : str
            Subject label (e.g., ``'sub-01'``).
        ses_label : str
            Session label (e.g., ``'ses-01'`` or ``'none'``).
        modalities : dict
            Dictionary with ``'dwi'``, ``'anat'``, ``'fmap'`` keys,
            each containing a list of file dicts.
        output_dir : Path
            Derivatives output directory root.
        pipeline_type : str
            Pipeline type.
        b0_threshold : int
            B0 threshold.
        denoise_method : str
            Denoising method.
        brain_mask_method : str
            Brain masking method.
        dti_fit : bool
            Whether to fit DTI.
        tracking : bool
            Whether to run tracking.

        Returns
        -------
        dict
            Dictionary with output paths and processing status for each
            stage.
        """
        dwi_files = modalities.get("dwi", [])
        anat_files = modalities.get("anat", [])

        if not dwi_files:
            logger.warning(
                f"No DWI files found for {sub_label}/{ses_label}, skipping."
            )
            return {"status": "skipped", "reason": "no DWI files"}

        # Build output directory structure
        if ses_label and ses_label != "none":
            out_dwi_dir = output_dir / sub_label / ses_label / "dwi"
            out_anat_dir = output_dir / sub_label / ses_label / "anat"
            prefix = f"{sub_label}_{ses_label}"
        else:
            out_dwi_dir = output_dir / sub_label / "dwi"
            out_anat_dir = output_dir / sub_label / "anat"
            prefix = sub_label

        out_dwi_dir.mkdir(parents=True, exist_ok=True)

        result = {"status": "completed", "outputs": {}}

        # Use the first DWI file (most common case: single DWI run)
        dwi_info = dwi_files[0]
        dwi_path = dwi_info["path"]
        bval_path = dwi_info.get("bval")
        bvec_path = dwi_info.get("bvec")

        logger.info(f"  DWI input: {dwi_path}")

        if not bval_path or not bvec_path:
            logger.warning(
                f"Missing bval/bvec for {dwi_path}. Some stages may be "
                "skipped."
            )

        # ---- Stage 1: Denoising ----
        if pipeline_type in ("full", "denoise"):
            denoised_path = out_dwi_dir / f"{prefix}_desc-denoised_dwi.nii.gz"
            result["outputs"]["denoised"] = str(denoised_path)

            try:
                self._run_denoise(
                    dwi_path=dwi_path,
                    bval_path=bval_path,
                    output_path=str(denoised_path),
                    method=denoise_method,
                    b0_threshold=b0_threshold,
                )
                logger.info(f"  Denoised output: {denoised_path}")
                # Use denoised data for subsequent stages
                current_dwi = str(denoised_path)
            except Exception as e:
                logger.error(f"  Denoising failed: {e}")
                result["outputs"]["denoised_error"] = str(e)
                current_dwi = dwi_path
        else:
            current_dwi = dwi_path

        # ---- Stage 2: Brain masking ----
        if pipeline_type in ("full", "mask"):
            mask_path = out_dwi_dir / f"{prefix}_desc-brain_mask.nii.gz"
            result["outputs"]["brain_mask"] = str(mask_path)

            # If we also have anat data, note the anat mask path
            if anat_files:
                out_anat_dir.mkdir(parents=True, exist_ok=True)
                anat_mask_path = (
                    out_anat_dir / f"{prefix}_desc-brain_mask.nii.gz"
                )
                result["outputs"]["anat_brain_mask"] = str(anat_mask_path)

            try:
                self._run_brain_mask(
                    input_path=current_dwi,
                    output_path=str(mask_path),
                    method=brain_mask_method,
                    bval_path=bval_path,
                    b0_threshold=b0_threshold,
                )
                logger.info(f"  Brain mask output: {mask_path}")
            except Exception as e:
                logger.error(f"  Brain masking failed: {e}")
                result["outputs"]["brain_mask_error"] = str(e)

        # ---- Stage 3: DTI fitting ----
        if pipeline_type in ("full", "dti") and dti_fit:
            if not bval_path or not bvec_path:
                logger.warning(
                    "Cannot run DTI fitting without bval/bvec files."
                )
            else:
                fa_path = out_dwi_dir / f"{prefix}_model-DTI_FA.nii.gz"
                md_path = out_dwi_dir / f"{prefix}_model-DTI_MD.nii.gz"
                rd_path = out_dwi_dir / f"{prefix}_model-DTI_RD.nii.gz"
                ad_path = out_dwi_dir / f"{prefix}_model-DTI_AD.nii.gz"
                rgb_path = (
                    out_dwi_dir / f"{prefix}_model-DTI_desc-color_FA.nii.gz"
                )

                result["outputs"]["FA"] = str(fa_path)
                result["outputs"]["MD"] = str(md_path)
                result["outputs"]["RD"] = str(rd_path)
                result["outputs"]["AD"] = str(ad_path)
                result["outputs"]["color_FA"] = str(rgb_path)

                try:
                    mask_for_dti = result["outputs"].get("brain_mask")
                    self._run_dti(
                        dwi_path=current_dwi,
                        bval_path=bval_path,
                        bvec_path=bvec_path,
                        mask_path=mask_for_dti,
                        fa_path=str(fa_path),
                        md_path=str(md_path),
                        rd_path=str(rd_path),
                        ad_path=str(ad_path),
                        rgb_path=str(rgb_path),
                        b0_threshold=b0_threshold,
                    )
                    logger.info(f"  DTI outputs in: {out_dwi_dir}")
                except Exception as e:
                    logger.error(f"  DTI fitting failed: {e}")
                    result["outputs"]["dti_error"] = str(e)

        logger.info(f"  {sub_label}/{ses_label} processing complete.")
        return result

    # ------------------------------------------------------------------
    #  Processing stage helpers
    # ------------------------------------------------------------------

    def _run_denoise(
        self, dwi_path, bval_path, output_path, method, b0_threshold
    ):
        """Run denoising on a DWI volume.

        Parameters
        ----------
        dwi_path : str
            Path to the input DWI NIfTI file.
        bval_path : str or None
            Path to the bval file.
        output_path : str
            Path for the denoised output.
        method : str
            Denoising method name.
        b0_threshold : int
            B0 threshold value.
        """
        data, affine, img = load_nifti(dwi_path, return_img=True)
        logger.info(
            f"    Denoising with method='{method}', shape={data.shape}"
        )

        if method == "patch2self":
            from dipy.denoise.patch2self import patch2self

            if bval_path is None:
                raise ValueError(
                    "patch2self denoising requires a bval file."
                )
            bvals, _ = read_bvals_bvecs(bval_path, None)
            denoised = patch2self(data, bvals, b0_threshold=b0_threshold)

        elif method == "mppca":
            from dipy.denoise.localpca import mppca

            denoised, _ = mppca(data, return_sigma=True)

        elif method == "lpca":
            from dipy.core.gradients import gradient_table
            from dipy.denoise.localpca import localpca
            from dipy.denoise.pca_noise_estimate import pca_noise_estimate

            if bval_path is None:
                raise ValueError("LPCA denoising requires a bval file.")
            bvals, bvecs = read_bvals_bvecs(bval_path, None)
            sigma = pca_noise_estimate(
                data, gradient_table(bvals, bvecs), correct_bias=True
            )
            denoised = localpca(data, sigma)

        elif method == "nlmeans":
            from dipy.denoise.nlmeans import nlmeans
            from dipy.denoise.noise_estimate import estimate_sigma

            sigma = estimate_sigma(data)
            denoised = nlmeans(data, sigma=sigma)

        else:
            raise ValueError(
                f"Unknown denoise method '{method}'. "
                "Choose from: patch2self, mppca, lpca, nlmeans."
            )

        save_nifti(output_path, denoised, affine, hdr=img.header)

    def _run_brain_mask(
        self, input_path, output_path, method, bval_path, b0_threshold
    ):
        """Run brain masking on a volume.

        Parameters
        ----------
        input_path : str
            Path to the input NIfTI file.
        output_path : str
            Path for the brain mask output.
        method : str
            Brain masking method name.
        bval_path : str or None
            Path to the bval file (needed for median_otsu with 4D data).
        b0_threshold : int
            B0 threshold value.
        """
        import numpy as np

        data, affine, img = load_nifti(input_path, return_img=True)
        logger.info(
            f"    Brain masking with method='{method}', shape={data.shape}"
        )

        if method == "median_otsu":
            from dipy.segment.mask import median_otsu

            vol_idx = None
            if data.ndim == 4 and bval_path is not None:
                bvals, _ = read_bvals_bvecs(bval_path, None)
                vol_idx = np.where(bvals <= b0_threshold)[0]

            _, mask = median_otsu(data, vol_idx=vol_idx)

        elif method == "synthseg":
            from dipy.nn.synthseg import SynthSeg

            synthseg = SynthSeg(verbose=True)
            vol = data[..., 0] if data.ndim == 4 else data
            _, _, mask = synthseg.predict(vol, affine)

        elif method == "evac":
            from dipy.nn.evac import EVACPlus

            evac = EVACPlus()
            vol = data[..., 0] if data.ndim == 4 else data
            mask = evac.predict(vol, affine)

        else:
            raise ValueError(
                f"Unknown brain_mask_method '{method}'. "
                "Choose from: synthseg, median_otsu, evac."
            )

        save_nifti(output_path, mask.astype(np.float64), affine)

    def _run_dti(
        self,
        dwi_path,
        bval_path,
        bvec_path,
        mask_path,
        fa_path,
        md_path,
        rd_path,
        ad_path,
        rgb_path,
        b0_threshold,
    ):
        """Run DTI model fitting.

        Parameters
        ----------
        dwi_path : str
            Path to the DWI NIfTI file.
        bval_path : str
            Path to the bval file.
        bvec_path : str
            Path to the bvec file.
        mask_path : str or None
            Path to a brain mask. If None, no mask is applied.
        fa_path : str
            Output path for FA map.
        md_path : str
            Output path for MD map.
        rd_path : str
            Output path for RD map.
        ad_path : str
            Output path for AD map.
        rgb_path : str
            Output path for color FA map.
        b0_threshold : int
            B0 threshold value.
        """
        import numpy as np

        from dipy.core.gradients import gradient_table
        from dipy.reconst.dti import (
            TensorModel,
            axial_diffusivity,
            color_fa,
            fractional_anisotropy,
            mean_diffusivity,
            radial_diffusivity,
        )

        data, affine = load_nifti(dwi_path)
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)

        logger.info(f"    DTI fitting, data shape={data.shape}")

        mask_data = None
        if mask_path and Path(mask_path).is_file():
            mask_data, _ = load_nifti(mask_path)
            mask_data = mask_data.astype(bool)

        tenmodel = TensorModel(gtab)

        if mask_data is not None:
            tenfit = tenmodel.fit(data, mask=mask_data)
        else:
            tenfit = tenmodel.fit(data)

        fa = fractional_anisotropy(tenfit.evals)
        fa = np.clip(fa, 0, 1)
        md = mean_diffusivity(tenfit.evals)
        ad = axial_diffusivity(tenfit.evals)
        rd = radial_diffusivity(tenfit.evals)
        cfa = color_fa(fa, tenfit.evecs)

        save_nifti(fa_path, fa.astype(np.float32), affine)
        save_nifti(md_path, md.astype(np.float32), affine)
        save_nifti(ad_path, ad.astype(np.float32), affine)
        save_nifti(rd_path, rd.astype(np.float32), affine)
        save_nifti(rgb_path, cfa.astype(np.float32), affine)
