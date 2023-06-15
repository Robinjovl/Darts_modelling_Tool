#ifndef VTKSNAPSHOTTER_H_ 
#define VTKSNAPSHOTTER_H_

#ifdef WITH_VTK

#include <string>
#include <unordered_map>

#include "mesh.h"

#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkPoints.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include <vtkTriangle.h>
#include <vtkQuad.h>
#include <vtkTetra.h>
#include <vtkHexahedron.h>
#include <vtkWedge.h>
#include <vtkPyramid.h>

namespace mesh
{
	class VTKSnapshotter
	{
	protected:
		const std::string prefix;
		std::string pattern;

		Mesh* mesh;

		std::string replace(std::string filename, std::string from, std::string to)
		{
			size_t start_pos = 0;
			while ((start_pos = filename.find(from, start_pos)) != std::string::npos)
			{
				filename.replace(start_pos, from.length(), to);
				start_pos += to.length();
			}
			return filename;
		};
		std::string get_file_name(index_t i)
		{
			std::string filename = pattern;
			return replace(filename, "%{STEP}", std::to_string(i));
		};
	public:
		VTKSnapshotter(const std::string _prefix = "snaps/") : prefix(_prefix)
		{
			pattern = prefix + "solution_%{STEP}.vtu";
		};
		~VTKSnapshotter() {};

		void set_mesh(Mesh* _mesh)
		{
			mesh = _mesh;
			if (mesh->name != "") pattern = prefix + mesh->name + "_%{STEP}.vtu";
		};
		void snapshot(int i)
		{
			auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
			auto points = vtkSmartPointer<vtkPoints>::New();

			// add points
			for (const auto& pt : mesh->nodes)
				points->InsertNextPoint(pt.x, pt.y, pt.z);
			grid->SetPoints(points);

			// add elements
			auto tetra = vtkSmartPointer<vtkTetra>::New();
			auto hex = vtkSmartPointer<vtkHexahedron>::New();
			auto wedge = vtkSmartPointer<vtkWedge>::New();
			auto pyramid = vtkSmartPointer<vtkPyramid>::New();
			tetra->GetPointIds()->SetNumberOfIds(4);
			hex->GetPointIds()->SetNumberOfIds(8);
			wedge->GetPointIds()->SetNumberOfIds(6);
			pyramid->GetPointIds()->SetNumberOfIds(5);

			auto tag = vtkSmartPointer<vtkIntArray>::New();
			tag->SetName("tag");

			auto elems = vtkSmartPointer<vtkCellArray>::New();
			for (const auto& elem : mesh->elems)
			{
				if (elem.type == TETRA)
				{
					for (uint8_t i = 0; i < elem.n_pts; i++)
						tetra->GetPointIds()->SetId(i, mesh->elem_nodes[elem.pts_offset + i]);
					grid->InsertNextCell(tetra->GetCellType(), tetra->GetPointIds());
				}
				else if (elem.type == HEX)
				{
					for (uint8_t i = 0; i < elem.n_pts; i++)
						hex->GetPointIds()->SetId(i, mesh->elem_nodes[elem.pts_offset + i]);
					grid->InsertNextCell(hex->GetCellType(), hex->GetPointIds());
				}
				else if (elem.type == PRISM)
				{
					for (uint8_t i = 0; i < elem.n_pts; i++)
						wedge->GetPointIds()->SetId(i, mesh->elem_nodes[elem.pts_offset + i]);
					grid->InsertNextCell(wedge->GetCellType(), wedge->GetPointIds());
				}
				else if (elem.type == PYRAMID)
				{
					for (uint8_t i = 0; i < elem.n_pts; i++)
						pyramid->GetPointIds()->SetId(i, mesh->elem_nodes[elem.pts_offset + i]);
					grid->InsertNextCell(pyramid->GetCellType(), pyramid->GetPointIds());
				}

				// cell data
				tag->InsertNextValue(elem.tag);
			}

			vtkCellData* cd = grid->GetCellData();
			cd->AddArray(tag);

			auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
			writer->SetFileName(get_file_name(i).c_str());
			writer->SetInputData(grid);
			writer->Write();
		};
	};
};

#endif /* WITH_VTK */

#endif /* VTKSNAPSHOTTER_H_ */