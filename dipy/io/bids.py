"""Lightweight BIDS layout parser for DIPY.

This module provides BIDS directory discovery without requiring pybids.
It parses standard BIDS directory structures to find diffusion MRI,
anatomical, and fieldmap files for each subject and session.

Notes
-----
This parser handles the standard BIDS naming convention::

    sub-XX/[ses-YY/]{dwi,anat,fmap}/sub-XX[_ses-YY]_*.<ext>

Only files with recognized suffixes are returned (e.g., ``_dwi.nii.gz``,
``_T1w.nii.gz``).
"""

import json
from pathlib import Path
import re

from dipy.utils.logging import logger


def is_bids_dir(path):
    """Check whether a directory looks like a BIDS dataset.

    A directory is considered BIDS if it contains a
    ``dataset_description.json`` file at its root.

    Parameters
    ----------
    path : str or Path
        Path to the candidate BIDS directory.

    Returns
    -------
    bool
        True if the directory contains ``dataset_description.json``.
    """
    p = Path(path)
    return p.is_dir() and (p / "dataset_description.json").is_file()


def _parse_bids_filename(fname):
    """Extract BIDS entities from a filename.

    Parameters
    ----------
    fname : str
        The filename (stem only, no directory) to parse.

    Returns
    -------
    dict
        Dictionary of entity key-value pairs (e.g., ``{'sub': '01',
        'ses': '02', 'suffix': 'dwi'}``).
    """
    entities = {}
    # Match key-value pairs like sub-01, ses-02, acq-multiband, etc.
    for match in re.finditer(r"([a-zA-Z]+)-([a-zA-Z0-9]+)", fname):
        entities[match.group(1)] = match.group(2)

    # Extract the suffix (last underscore-separated element before extension)
    # e.g., sub-01_ses-02_dwi.nii.gz -> suffix = "dwi"
    stem = fname.split(".")[0]
    parts = stem.split("_")
    if parts:
        last = parts[-1]
        # If the last part doesn't contain a dash, it's the suffix
        if "-" not in last:
            entities["suffix"] = last

    return entities


def _find_files(directory, pattern="*.nii*"):
    """Recursively find files matching a glob pattern.

    Parameters
    ----------
    directory : Path
        The directory to search.
    pattern : str, optional
        Glob pattern to match.

    Returns
    -------
    list of Path
        Sorted list of matching file paths.
    """
    return sorted(directory.rglob(pattern))


def _collect_sidecars(nifti_path):
    """Find sidecar files (.bval, .bvec, .json) for a NIfTI file.

    Parameters
    ----------
    nifti_path : Path
        Path to a NIfTI file.

    Returns
    -------
    dict
        Dictionary with keys ``'bval'``, ``'bvec'``, ``'json'`` mapping to
        file paths if they exist, or None otherwise.
    """
    # Strip all extensions (handles .nii.gz)
    stem = nifti_path.name
    for ext in [".nii.gz", ".nii"]:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    parent = nifti_path.parent
    sidecars = {}
    for key, suffix in [("bval", ".bval"), ("bvec", ".bvec"), ("json", ".json")]:
        sidecar = parent / (stem + suffix)
        sidecars[key] = str(sidecar) if sidecar.is_file() else None

    return sidecars


def _discover_subjects(bids_dir):
    """Find all subject directories in a BIDS dataset.

    Parameters
    ----------
    bids_dir : Path
        Root of the BIDS dataset.

    Returns
    -------
    list of str
        Sorted list of subject labels (e.g., ``['01', '02']``).
    """
    subjects = []
    for d in sorted(bids_dir.iterdir()):
        if d.is_dir() and d.name.startswith("sub-"):
            subjects.append(d.name[4:])  # strip 'sub-' prefix
    return subjects


def _discover_sessions(subject_dir):
    """Find all session directories for a given subject.

    Parameters
    ----------
    subject_dir : Path
        Path to the subject directory (e.g., ``bids_dir/sub-01``).

    Returns
    -------
    list of str or list containing None
        Sorted list of session labels, or ``[None]`` if there are no
        session directories.
    """
    sessions = []
    for d in sorted(subject_dir.iterdir()):
        if d.is_dir() and d.name.startswith("ses-"):
            sessions.append(d.name[4:])  # strip 'ses-' prefix
    return sessions if sessions else [None]


def _collect_modality_files(modality_dir, suffixes):
    """Collect NIfTI files from a modality directory that match given suffixes.

    Parameters
    ----------
    modality_dir : Path
        Path to a modality directory (e.g., ``sub-01/dwi/``).
    suffixes : list of str
        BIDS suffixes to match (e.g., ``['dwi']``, ``['T1w', 'T2w']``).

    Returns
    -------
    list of dict
        Each dict contains ``'path'`` (str), ``'entities'`` (dict of BIDS
        entities), and sidecar keys ``'bval'``, ``'bvec'``, ``'json'``.
    """
    if not modality_dir.is_dir():
        return []

    files = []
    for nifti in sorted(modality_dir.glob("*.nii*")):
        entities = _parse_bids_filename(nifti.name)
        if entities.get("suffix") in suffixes:
            entry = {
                "path": str(nifti),
                "entities": entities,
            }
            entry.update(_collect_sidecars(nifti))
            files.append(entry)
    return files


def load_bids_layout(bids_dir, participant_label=None, session_label=None):
    """Parse a BIDS directory and return a structured layout of files.

    This function walks the BIDS directory tree to discover subjects,
    sessions, and relevant neuroimaging files (DWI, anatomical, fieldmap)
    without requiring the pybids package.

    Parameters
    ----------
    bids_dir : str or Path
        Path to the root of the BIDS dataset. Must contain a
        ``dataset_description.json`` file.
    participant_label : list of str, optional
        List of participant labels to include (e.g., ``['01', '02']`` or
        ``['sub-01', 'sub-02']``). If None, all subjects are included.
    session_label : list of str, optional
        List of session labels to include (e.g., ``['01', '02']`` or
        ``['ses-01', 'ses-02']``). If None, all sessions are included.

    Returns
    -------
    dict
        A nested dictionary with the following structure::

            {
                'dataset_description': dict,
                'subjects': {
                    'sub-01': {
                        'ses-01': {  # or 'none' if no sessions
                            'dwi': [{'path': ..., 'bval': ..., 'bvec': ...,
                                      'json': ..., 'entities': {...}}, ...],
                            'anat': [{'path': ..., 'json': ...,
                                      'entities': {...}}, ...],
                            'fmap': [{'path': ..., 'json': ...,
                                      'entities': {...}}, ...],
                        },
                        ...
                    },
                    ...
                }
            }

    Raises
    ------
    FileNotFoundError
        If ``bids_dir`` does not exist or does not contain
        ``dataset_description.json``.

    Examples
    --------
    >>> layout = load_bids_layout('/data/my_study')
    >>> for sub, sessions in layout['subjects'].items():
    ...     for ses, modalities in sessions.items():
    ...         for dwi_file in modalities['dwi']:
    ...             print(dwi_file['path'])
    """
    bids_dir = Path(bids_dir)

    if not bids_dir.is_dir():
        raise FileNotFoundError(f"BIDS directory not found: {bids_dir}")

    desc_file = bids_dir / "dataset_description.json"
    if not desc_file.is_file():
        raise FileNotFoundError(
            f"Not a valid BIDS dataset (missing dataset_description.json): "
            f"{bids_dir}"
        )

    with open(desc_file) as f:
        dataset_description = json.load(f)

    # Normalize participant labels: strip 'sub-' prefix if present
    if participant_label is not None:
        participant_label = [
            p[4:] if p.startswith("sub-") else p for p in participant_label
        ]

    # Normalize session labels: strip 'ses-' prefix if present
    if session_label is not None:
        session_label = [
            s[4:] if s.startswith("ses-") else s for s in session_label
        ]

    all_subjects = _discover_subjects(bids_dir)

    # Filter subjects
    if participant_label is not None:
        subjects = [s for s in all_subjects if s in participant_label]
        missing = set(participant_label) - set(subjects)
        if missing:
            logger.warning(
                f"Requested participants not found in dataset: "
                f"{['sub-' + m for m in missing]}"
            )
    else:
        subjects = all_subjects

    layout = {
        "dataset_description": dataset_description,
        "subjects": {},
    }

    for sub in subjects:
        sub_label = f"sub-{sub}"
        sub_dir = bids_dir / sub_label
        sessions = _discover_sessions(sub_dir)

        # Filter sessions
        if session_label is not None:
            sessions = [s for s in sessions if s in session_label]

        layout["subjects"][sub_label] = {}

        for ses in sessions:
            if ses is not None:
                ses_label = f"ses-{ses}"
                base_dir = sub_dir / ses_label
            else:
                ses_label = "none"
                base_dir = sub_dir

            dwi_files = _collect_modality_files(
                base_dir / "dwi", ["dwi"]
            )
            anat_files = _collect_modality_files(
                base_dir / "anat", ["T1w", "T2w", "FLAIR"]
            )
            fmap_files = _collect_modality_files(
                base_dir / "fmap",
                ["epi", "magnitude1", "magnitude2", "phasediff", "fieldmap"],
            )

            layout["subjects"][sub_label][ses_label] = {
                "dwi": dwi_files,
                "anat": anat_files,
                "fmap": fmap_files,
            }

            n_dwi = len(dwi_files)
            n_anat = len(anat_files)
            n_fmap = len(fmap_files)
            logger.debug(
                f"  {sub_label}/{ses_label}: "
                f"{n_dwi} DWI, {n_anat} anat, {n_fmap} fmap files"
            )

    n_subs = len(layout["subjects"])
    logger.info(f"BIDS layout: found {n_subs} subject(s) in {bids_dir}")

    return layout


def write_derivative_description(output_dir, source_dataset=None):
    """Write a BIDS-compliant ``dataset_description.json`` for derivatives.

    Parameters
    ----------
    output_dir : str or Path
        Path to the derivatives output directory.
    source_dataset : dict, optional
        The source dataset's ``dataset_description.json`` contents. Used to
        populate the ``GeneratedBy`` and ``SourceDatasets`` fields.

    Returns
    -------
    Path
        Path to the written ``dataset_description.json``.
    """
    from dipy import __version__ as dipy_version

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    description = {
        "Name": "DIPY Auto Pipeline",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "dipy_auto",
                "Version": dipy_version,
                "CodeURL": "https://github.com/dipy/dipy",
            }
        ],
    }

    if source_dataset:
        source_name = source_dataset.get("Name", "Unknown")
        source_url = source_dataset.get("URL", source_dataset.get("bids_dir", ""))
        source_entry = {"Name": source_name}
        if source_url:
            source_entry["URL"] = str(source_url)
        description["SourceDatasets"] = [source_entry]

    desc_path = output_dir / "dataset_description.json"
    with open(desc_path, "w") as f:
        json.dump(description, f, indent=2)
        f.write("\n")

    logger.info(f"Wrote derivative dataset_description.json to {desc_path}")
    return desc_path
