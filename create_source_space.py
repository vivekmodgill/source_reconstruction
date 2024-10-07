import mne

# Set paths and subject ID
subjects_dir = '/path/to/freesurfer/subjects'
subject      = 'subject_id'

# Set up source space with oct6 spacing (6th subdivision of icosahedron)
src = mne.setup_source_space(subject, spacing="oct6", add_dist="patch", subjects_dir=subjects_dir)
print(f"Source space details:\n{src}")

# Save the source space to a compressed FIF file
src_fname = 'src.fif.gz'
mne.write_source_spaces(src_fname, src, overwrite=True)

# Create BEM model using watershed algorithm
mne.bem.make_watershed_bem(subject=subject, subjects_dir=subjects_dir, overwrite=True, volume='T1')

# Define conductivity for single layer BEM (use three layers if needed)
conductivity = (0.3,)  # Single layer (brain tissue)

# Uncomment for three-layer model (scalp, skull, brain)
# conductivity = (0.3, 0.006, 0.3)

# Create BEM model with the specified conductivity and ico4 (low-res) subdivision
bem_model = mne.make_bem_model(subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir)

# Compute the BEM solution
bem_solution = mne.make_bem_solution(bem_model)

# Save the BEM solution to a FIF file
bem_fname = 'bem-sol.fif'
mne.write_bem_solution(bem_fname, bem_solution, overwrite=False)

print(f"BEM solution saved to: {bem_fname}")
