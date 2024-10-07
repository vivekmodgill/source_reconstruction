import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import create_ecg_epochs, create_eog_epochs, ICA
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

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

for subject_id in subjects_list:
    try:
        # Define paths for the current subject
        trans_path   = os.path.join('/path/to/trans/files/', f'{subject_id}-trans.fif')
        raw_fif_path = os.path.join('/path/to/raw/data/', f'{subject_id}.fif')
        
        # Load raw data
        raw = mne.io.read_raw_fif(raw_fif_path, preload=True, verbose='error')
        raw.filter(0.1, None, fir_design='firwin')  # High-pass filter

        # ICA for artifact removal
        ica   = ICA(n_components=0.95, method='fastica')
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')
        ica.fit(raw, picks=picks, decim=3, reject=dict(mag=3000e-15, grad=3000e-13, eeg=100e-6, eog=200e-6))

        # ECG and EOG artifact detection
        ecg_epochs  = create_ecg_epochs(raw, tmin=-.5, tmax=.5, picks=picks)
        ecg_inds, _ = ica.find_bads_ecg(ecg_epochs, method='ctps')
        ica.exclude.extend(ecg_inds[:3])  # Limit to 3 ECG components

        eog_inds, _ = ica.find_bads_eog(raw)
        ica.exclude.extend(eog_inds[:1])  # Limit to 1 EOG component

        # Apply ICA
        raw_ica = ica.apply(raw)
        raw_ica.crop(0, None).pick_types(meg='mag')

        # Compute noise covariance matrix
        cov = mne.make_ad_hoc_cov(raw_ica.info)

        # Epoching data
        events = mne.make_fixed_length_events(raw_ica, duration=5.0)
        epochs = mne.Epochs(raw_ica, events=events, tmin=0, tmax=5.0, baseline=None, reject=None, preload=True)

        # Source reconstruction
        src = mne.read_source_spaces(src_path)
        fwd = mne.make_forward_solution(epochs.info, trans=trans_path, src=src, bem=bem_path)
        inv = make_inverse_operator(epochs.info, fwd, cov)

        # Apply inverse operator
        snr     = 3.0
        lambda2 = 1.0 / snr ** 2
        method  = "sLORETA"
        stcs    = apply_inverse_epochs(epochs, inv, lambda2, method, pick_ori="normal", return_generator=True)

        # Extract label time series
        labels   = mne.read_labels_from_annot(subject=subject, parc='aparc', subjects_dir=subjects_dir)
        label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip')

        # Save the extracted time courses
        output_path = os.path.join(ica_corrected_dir, f'{subject_id}.npy')
        np.save(output_path, label_ts)

        print(f'Processing complete for {subject_id}')
    
    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
