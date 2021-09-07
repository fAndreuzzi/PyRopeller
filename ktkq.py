import numpy as np
import vtk


def field_on_boundary(smithers_time_dict, boundary_key, field_key):
    faces_dict = smithers_time_dict["boundary"][boundary_key]["faces"]
    face_idxes = faces_dict["faces_indexes"]
    face_normals = faces_dict["normal"]

    # the index of the cell which owns each face which belongs to propellerTip
    outer_cells_labels = list(
        map(smithers_time_dict["face_owner_cell"].__getitem__, face_idxes)
    )

    # we use the index of the cell to obtain the value of the pressure at
    # the center of that cell, for each cell which touches the boundary
    return np.asarray(smithers_time_dict["fields"][field_key][1])[
        outer_cells_labels
    ]


def normals_to_boundary(smithers_time_dict, boundary_key):
    return smithers_time_dict["boundary"][boundary_key]["faces"]["normal"]


def t(smithers_time_dict):
    pressure = field_on_boundary(smithers_time_dict, b'propellerTip', 'p')
    face_normals = normals_to_boundary(smithers_time_dict, b'propellerTip')
    return np.sum(pressure[:, None] * face_normals, axis=0)


def t_time_progression(smithers_dict):
    return [
        t(smithers_dict[time]) for time in sorted(list(smithers_dict.keys()))
    ]

