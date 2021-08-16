import numpy as np

def t(smithers_time_dict):
    faces_dict = smithers_time_dict['boundary'][b"propellerTip"]['faces']
    face_idxes = faces_dict['faces_indexes']
    face_areas = faces_dict['area']
    face_normals = faces_dict['normal']
    # the index of the cell which owns each face which belongs to propellerTip
    outer_cells_labels = list(map(smithers_time_dict['face_ownert_cell'].__getitem__, face_idxes))
    # we use the index of the cell to obtain the value of the pressure at
    # the center of that cell, for each cell which touches the boundary
    pressure = np.asarray(smithers_time_dict['fields']['p'][1])[outer_cells_labels]
    return np.sum(pressure[:,None] * face_normals, axis=0)
