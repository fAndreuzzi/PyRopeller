import numpy as np
from smithers.io.vtkhandler import VTKHandler
from vtk import vtkPolyDataReader, vtkPolyDataWriter, vtkTriangleFilter
import os


def normals_and_area(vtk_data):
    def nrm(points):
        v1 = vtk_data["points"][points[1]] - vtk_data["points"][points[0]]
        v2 = vtk_data["points"][points[2]] - vtk_data["points"][points[0]]

        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)

        return (n / norm, abs(norm) / 2)

    return np.apply_along_axis(nrm, 0, vtk_data['cells'])


def triangulate_vtk(vtk_path):
    handler = vtkPolyDataReader()
    handler.SetFileName(vtk_path)
    handler.Update()
    polydata = handler.GetOutput()
    filter = vtkTriangleFilter()
    filter.SetInputData(polydata)
    filter.Update()
    return filter.GetOutput()


def t(vtk_path):
    handler = VTKHandler(vtkPolyDataReader, vtkPolyDataWriter)
    triangulated_vtk_data = triangulate_vtk(vtk_path)
    data = handler.parse(triangulated_vtk_data)

    pressure = data["cell_data"]["p"]
    na = normals_and_area(data)
    n = na[:,0]
    a = na[:,1]

    return np.sum(pressure * n * a, axis=0)


def t_time_progression(vtk_folder_path):
    return list(
        map(
            t,
            # we add the leading part to make the path absolute
            map(
                lambda s: vtk_folder_path + "/" + s,
                os.listdir(vtk_folder_path),
            ),
        )
    )
