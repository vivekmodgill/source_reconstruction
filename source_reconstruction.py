import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import create_ecg_epochs, create_eog_epochs, ICA
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from autoreject import get_rejection_threshold
import traceback

# Define constants (paths need to be set)
subjects_dir      = '/path/to/freesurfer/subjects/'
subject           = 'subject id'
bem_path          = os.path.join(subjects_dir, subject, 'bem', 'bem_sol.fif')
src_path          = os.path.join(subjects_dir, subject, 'bem', 'src.fif.gz')
ica_corrected_dir = '/path/to/save/ica_corrected_data/'

# List of subjects (to be processed)
subjects_list = [
    # Add your subject IDs here
    'subject_1', 'subject_2', 'subject_3', '...'
]

for i in subjects_list:
    try:
        # Paths
        trans_path   = os.path.join('/vol/specs04/vivek/camcan/fiducials', f'{i}-trans.fif')
        raw_fif_path = os.path.join('/vol/specs04/vivek/camcan/CamCAN', f'{i}.fif')

        # Load raw data
        raw = mne.io.read_raw_fif(raw_fif_path, preload=True, verbose='error')
        raw.filter(0.1, None, fir_design='firwin')  # High-pass filter

        # Epoch the data
        events = mne.make_fixed_length_events(raw, duration=5.0)
        epochs = mne.Epochs(raw, events=events, tmin=0, tmax=5.0, baseline=None, reject=None, preload=True)

        # Get rejection threshold from epochs
        reject = get_rejection_threshold(epochs)
        epochs = mne.Epochs(raw, events=events, tmin=0, tmax=5.0, baseline=None, reject=reject, preload=True)

        # ICA for artifact removal
        picks = mne.pick_types(epochs.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')
        ica   = ICA(n_components=0.95, method='fastica')
        ica.fit(epochs, picks=picks, decim=3)

        # ECG and EOG artifact detection
        ecg_epochs  = create_ecg_epochs(raw, tmin=-0.5, tmax=0.5, picks=picks)
        ecg_inds, _ = ica.find_bads_ecg(ecg_epochs, method='ctps')
        ica.exclude.extend(ecg_inds[:3])  # Limiting to 3 ECG components

        eog_inds, _ = ica.find_bads_eog(raw)
        ica.exclude.extend(eog_inds[:1])  # Limiting to 1 EOG component

        # Apply ICA to remove artifacts
        epochs_ica = ica.apply(epochs)

        # Re-filter after ICA
        epochs_ica.crop(0, None).pick_types(meg='mag')
        # Compute noise covariance matrix
        cov = mne.make_ad_hoc_cov(epochs_ica.info)

        # Source reconstruction
        src = mne.read_source_spaces(src_path)
        fwd = mne.make_forward_solution(epochs_ica.info, trans=trans_path, src=src, bem=bem_path)
        inv = make_inverse_operator(epochs_ica.info, fwd, cov)

        # Apply inverse operator and extract source time courses
        snr     = 3.0
        lambda2 = 1.0 / snr ** 2
        method  = "sLORETA"
        stcs    = apply_inverse_epochs(epochs_ica, inv, lambda2, method, pick_ori="normal", return_generator=True)

        # Extract label time series
        labels   = mne.read_labels_from_annot(subject='colin27', parc='aparc', subjects_dir=subjects_dir)
        label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip')

        # Save the extracted time courses
        output_path = os.path.join(ica_corrected_dir, f'{i}.npy')
        np.save(output_path, label_ts)

        print(f'Processing complete for {i}')

    except Exception as e:
        print(f"Error processing {i}: {e}")
        traceback.print_exc()  # This will print the full traceback of the error
