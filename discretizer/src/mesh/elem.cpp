#include "elem.h"

using namespace mesh;
using linalg::Vector3;


void Elem::calculate_centroid(const std::vector<Vector3>& nodes, const std::vector<index_t>& elem_nodes, Vector3 &c)
{
	// the centroid location on x,y,z direction
	double Cx = 0.0;
	double Cy = 0.0;
	double Cz = 0.0;
	
	// iterate among all points (nodes) in the element
	for (int i = this->pts_offset; i < this->pts_offset + this->n_pts; ++i) {
		int node_idx = elem_nodes[i]; // read the index of the element in the elem_nodes vector
		auto node = nodes[node_idx]; // read the element node (containing x,y,z coordinates) from the nodes vector

		// the centroid is the arithmetic avg of the coordinates
		Cx += node.x;
		Cy += node.y;
		Cz += node.z;
	}

	Cx /= this->n_pts;
	Cy /= this->n_pts;
	Cz /= this->n_pts;


    // put all of the elements nodes into a vector
    std::vector<Vector3> element_nodes;

    for (int i = this->pts_offset; i < this->pts_offset + this->n_pts; i++) {
        index_t node_idx = elem_nodes[i];
        Vector3 node = nodes[node_idx];
        element_nodes.push_back(node);
    }

    switch (this->type) {
    case TETRA: {
        //TODO: why volume calculated here? there is another function for it
        //this->volume = tetra_volume(element_nodes);
        break;
    }
    case LINE: {
        // the centroid of a line is its middle point
        for (auto i : element_nodes) {
            Cx += i.x / 2;
            Cy += i.y / 2;
            Cz += i.z / 2;
        }
        
        break;
    }
    case TRI: {
        for (auto i : element_nodes) {
            Cx += i.x / 3;
            Cy += i.y / 3;
            Cz += i.z / 3;
        }
        
        break;
    }
    case QUAD: {
        Vector3 corner1 = element_nodes[0] - element_nodes[1];
        Vector3 corner2 = element_nodes[0] - element_nodes[3];
        Vector3 corner3 = element_nodes[2] - element_nodes[1];
        Vector3 corner4 = element_nodes[2] - element_nodes[3];

        // volume is computed as area * "pseudo thickness"
        //this->volume = cell_area * 10e-4;
        break;
    }
    case HEX: {
        // split the hexahedron into five tetrahedrons and sum their volumes
        std::vector<Vector3> tetra1{ nodes[1], nodes[2], nodes[3], nodes[6] };
        std::vector<Vector3> tetra2{ nodes[1], nodes[6], nodes[3], nodes[7] };
        std::vector<Vector3> tetra3{ nodes[1], nodes[7], nodes[5], nodes[6] };
        std::vector<Vector3> tetra4{ nodes[1], nodes[7], nodes[5], nodes[4] };
        std::vector<Vector3> tetra5{ nodes[1], nodes[0], nodes[3], nodes[4] };
        //this->volume = tetra_volume(tetra1) + tetra_volume(tetra2) + tetra_volume(tetra3) + tetra_volume(tetra4) +
        //    tetra_volume(tetra5);
        break;
    }
    case PRISM: {
        // subdivide into 3 tetras, total volume is the sum of their individual volumes
        std::vector<Vector3> tetra1{ element_nodes[0], element_nodes[3], element_nodes[4], element_nodes[5] };
        std::vector<Vector3> tetra2{ element_nodes[0], element_nodes[1], element_nodes[4], element_nodes[5] };
        std::vector<Vector3> tetra3{ element_nodes[1], element_nodes[2], element_nodes[5], element_nodes[0] };
        //this->volume = tetra_volume(tetra1) + tetra_volume(tetra2) + tetra_volume(tetra3);
        break;
    }
    case PYRAMID: {

        // diagonals of the bottom face of rectangular pyramid
        Vector3 diag1 = element_nodes[0] - element_nodes[2];
        Vector3 diag2 = element_nodes[1] - element_nodes[3];

        // depending on which diagonal is longer, subdivide in different ways
        if (diag1.norm_sq() > diag2.norm_sq()) {
            // subdivide into two tetras, total volume is the sum of their volumes
            std::vector<Vector3> tetra1{ element_nodes[0], element_nodes[2], element_nodes[1], element_nodes[4] };
            std::vector<Vector3> tetra2{ element_nodes[0], element_nodes[2], element_nodes[3], element_nodes[4] };
            //this->volume = tetra_volume(tetra1) + tetra_volume(tetra2);
        }
        else {
            // subdivide into two tetras, total volume is the sum of their volumes
            std::vector<Vector3> tetra1{ element_nodes[0], element_nodes[1], element_nodes[3], element_nodes[4] };
            std::vector<Vector3> tetra2{ element_nodes[1], element_nodes[3], element_nodes[2], element_nodes[4] };
            //this->volume = tetra_volume(tetra1) + tetra_volume(tetra2);
        }

        break;
    }
    }

	// set the centroid value
	c = Vector3{ Cx, Cy, Cz };
}

Vector3 mesh::triangle_centroid_area(const std::vector<Vector3>& nodes, double* cell_area) {
    Vector3 corner1 = nodes[0] - nodes[1];
    Vector3 corner2 = nodes[0] - nodes[2];

    *cell_area = 0.5 * cross(corner1, corner2).norm();
    
    double Cx = 0.0;
    double Cy = 0.0;
    double Cz = 0.0;

    for (auto i : nodes) {
        Cx += i.x / 3;
        Cy += i.y / 3;
        Cz += i.z / 3;
    }

    // the centroid coordinates
    return Vector3{ Cx, Cy, Cz };
}

Vector3 mesh::tetra_centroid(const std::vector<Vector3>& nodes) {
    double Cx = 0.0;
    double Cy = 0.0;
    double Cz = 0.0;

    for (auto i : nodes) {
        Cx += i.x / 4;
        Cy += i.y / 4;
        Cz += i.z / 4;
    }

    return Vector3{ Cx, Cy, Cz };
}

double mesh::tetra_volume(const std::vector<Vector3>& nodes) {

	auto a = nodes[0];
	auto b = nodes[1];
	auto c = nodes[2];
	auto d = nodes[3];

	// get the edge vectors from all other vertices to d
	auto edge1 = a - d;
	auto edge2 = b - d;
	auto edge3 = c - d;

	return std::abs(linalg::dot(edge1, linalg::cross(edge2, edge3))) / 6.0;
}

void Elem::calculate_volume_and_centroid(const std::vector<Vector3>& nodes, const std::vector<index_t>& elem_nodes,
																				 value_t &volume, Vector3 &c){

    // put all of the elements nodes into a vector
    std::vector<Vector3> element_nodes;

    for (int i = this->pts_offset; i < this->pts_offset + this->n_pts; i++) {
        index_t node_idx = elem_nodes[i];
        Vector3 node = nodes[node_idx];
        element_nodes.push_back(node);
    }

    switch (this->type) {
    case TETRA: {
        volume = tetra_volume(element_nodes);
        c = tetra_centroid(element_nodes);
        break;
    }

    case LINE: {
        volume = (element_nodes[0] - element_nodes[1]).norm();

        double Cx = 0.0;
        double Cy = 0.0;
        double Cz = 0.0;

        // the centroid of a line is its middle point
        for (auto i : element_nodes) {
            Cx += i.x / 2;
            Cy += i.y / 2;
            Cz += i.z / 2;
        }
        c = Vector3{ Cx, Cy, Cz };
        break;
    }

    case TRI: {
        double cell_area = 0.0;

        // volume is computed as area * "pseudo thickness"
        c = triangle_centroid_area(element_nodes, &cell_area);
        volume = cell_area * 10e-4;
        break;
    }

    case QUAD: {
        Vector3 corner1 = element_nodes[0] - element_nodes[1];
        Vector3 corner2 = element_nodes[0] - element_nodes[3];
        Vector3 corner3 = element_nodes[2] - element_nodes[1];
        Vector3 corner4 = element_nodes[2] - element_nodes[3];
        
        std::vector<Vector3> nodes_triangle1 = { element_nodes[0], element_nodes[1], element_nodes[2] };
        std::vector<Vector3> nodes_triangle2 = { element_nodes[0], element_nodes[2], element_nodes[3] };

        double area_triangle1, area_triangle2;
        Vector3 centroid1 = triangle_centroid_area(nodes_triangle1, &area_triangle1);
        Vector3 centroid2 = triangle_centroid_area(nodes_triangle2, &area_triangle2);
           
        //Cx = (centroid1.x * area_triangle1 + centroid2.x * area_triangle2) / (area_triangle1 + area_triangle2);
        //Cy = (centroid1.y * area_triangle1 + centroid2.y * area_triangle2) / (area_triangle1 + area_triangle2);
        //Cz = (centroid1.z * area_triangle1 + centroid2.z * area_triangle2) / (area_triangle1 + area_triangle2);

        double Cx = 0.0;
        double Cy = 0.0;
        double Cz = 0.0;

        // iterate among all points (nodes) in the element
        for (int i = this->pts_offset; i < this->pts_offset + this->n_pts; ++i) {
            int node_idx = elem_nodes[i]; // read the index of the element in the elem_nodes vector
            auto node = nodes[node_idx]; // read the element node (containing x,y,z coordinates) from the nodes vector

            // the centroid is the arithmetic avg of the coordinates
            Cx += node.x;
            Cy += node.y;
            Cz += node.z;
        }

        Cx /= this->n_pts;
        Cy /= this->n_pts;
        Cz /= this->n_pts;

        c = Vector3{ Cx, Cy, Cz };

        double cell_area = 0.5 * cross(corner1, corner2).norm() + cross(corner3, corner4).norm();
        // volume is computed as area * "pseudo thickness"
        volume = cell_area * 10e-4;
        break;
    }

    case HEX: {

        // split the hexahedron into five tetrahedrons and sum their volumes
        std::vector<Vector3> tetra1{ element_nodes[0], element_nodes[1], element_nodes[2], element_nodes[5] };
        std::vector<Vector3> tetra2{ element_nodes[0], element_nodes[2], element_nodes[3], element_nodes[7] };
        std::vector<Vector3> tetra3{ element_nodes[6], element_nodes[5], element_nodes[7], element_nodes[2] };
        std::vector<Vector3> tetra4{ element_nodes[0], element_nodes[2], element_nodes[7], element_nodes[5] };
        std::vector<Vector3> tetra5{ element_nodes[0], element_nodes[5], element_nodes[7], element_nodes[4] };


        Vector3 centroid1 = tetra_centroid(tetra1);
        Vector3 centroid2 = tetra_centroid(tetra2);
        Vector3 centroid3 = tetra_centroid(tetra3);
        Vector3 centroid4 = tetra_centroid(tetra4);
        Vector3 centroid5 = tetra_centroid(tetra5);

        double volume1 = tetra_volume(tetra1);
        double volume2 = tetra_volume(tetra2);
        double volume3 = tetra_volume(tetra3);
        double volume4 = tetra_volume(tetra4);
        double volume5 = tetra_volume(tetra5);

        volume = 0;
        for (auto i : { volume1, volume2, volume3, volume4, volume5 }) {
            volume += i;
        }

        //Cx = (centroid1.x * volume1 + centroid2.x * volume2 + centroid3.x * volume3 + centroid4.x * volume4 + centroid5.x * volume5) / this->volume;
        //Cy = (centroid1.y * volume1 + centroid2.y * volume2 + centroid3.y * volume3 + centroid4.y * volume4 + centroid5.y * volume5) / this->volume;
        //Cz = (centroid1.z * volume1 + centroid2.z * volume2 + centroid3.z * volume3 + centroid4.z * volume4 + centroid5.z * volume5) / this->volume;

        double Cx = 0.0;
        double Cy = 0.0;
        double Cz = 0.0;

        // iterate among all points (nodes) in the element
        for (int i = this->pts_offset; i < this->pts_offset + this->n_pts; ++i) {
            int node_idx = elem_nodes[i]; // read the index of the element in the elem_nodes vector
            auto node = nodes[node_idx]; // read the element node (containing x,y,z coordinates) from the nodes vector

            // the centroid is the arithmetic avg of the coordinates
            Cx += node.x;
            Cy += node.y;
            Cz += node.z;
        }

        Cx /= this->n_pts;
        Cy /= this->n_pts;
        Cz /= this->n_pts;


        c = Vector3{ Cx, Cy, Cz };

        break;
    }

    case PRISM: {
        // subdivide into 3 tetras, total volume is the sum of their individual volumes
        std::vector<Vector3> tetra1{ element_nodes[0], element_nodes[3], element_nodes[4], element_nodes[5] };
        std::vector<Vector3> tetra2{ element_nodes[0], element_nodes[1], element_nodes[4], element_nodes[5] };
        std::vector<Vector3> tetra3{ element_nodes[1], element_nodes[2], element_nodes[5], element_nodes[0] };

        double volume1 = tetra_volume(tetra1);
        double volume2 = tetra_volume(tetra2);
        double volume3 = tetra_volume(tetra3);

        volume = 0;
        for (auto i : { volume1, volume2, volume3}) {
            volume += i;
        }

        //Cx = (centroid1.x * volume1 + centroid2.x * volume2 + centroid3.x * volume3) / this->volume;
        //Cy = (centroid1.y * volume1 + centroid2.y * volume2 + centroid3.y * volume3) / this->volume;
        //Cz = (centroid1.z * volume1 + centroid2.z * volume2 + centroid3.z * volume3) / this->volume;

        double Cx = 0.0;
        double Cy = 0.0;
        double Cz = 0.0;

        // iterate among all points (nodes) in the element
        for (int i = this->pts_offset; i < this->pts_offset + this->n_pts; ++i) {
            int node_idx = elem_nodes[i]; // read the index of the element in the elem_nodes vector
            auto node = nodes[node_idx]; // read the element node (containing x,y,z coordinates) from the nodes vector

            // the centroid is the arithmetic avg of the coordinates
            Cx += node.x;
            Cy += node.y;
            Cz += node.z;
        }

        Cx /= this->n_pts;
        Cy /= this->n_pts;
        Cz /= this->n_pts;

        c = Vector3{ Cx, Cy, Cz };
        break;
    }

    case PYRAMID: {

        // diagonals of the bottom face of rectangular pyramid
        Vector3 diag1 = element_nodes[0] - element_nodes[2];
        Vector3 diag2 = element_nodes[1] - element_nodes[3];

        // depending on which diagonal is longer, subdivide in different ways
        if (diag1.norm_sq() > diag2.norm_sq()) {
            // subdivide into two tetras, total volume is the sum of their volumes
            std::vector<Vector3> tetra1{ element_nodes[0], element_nodes[2], element_nodes[1], element_nodes[4] };
            std::vector<Vector3> tetra2{ element_nodes[0], element_nodes[2], element_nodes[3], element_nodes[4] };

            Vector3 centroid1 = tetra_centroid(tetra1);
            Vector3 centroid2 = tetra_centroid(tetra2);

            double volume1 = tetra_volume(tetra1);
            double volume2 = tetra_volume(tetra2);

            volume = tetra_volume(tetra1) + tetra_volume(tetra2);

            //Cx = (centroid1.x * volume1 + centroid2.x * volume2) / this->volume;
            //Cy = (centroid1.y * volume1 + centroid2.y * volume2) / this->volume;
            //Cz = (centroid1.z * volume1 + centroid2.z * volume2) / this->volume;

            double Cx = 0.0;
            double Cy = 0.0;
            double Cz = 0.0;

            // iterate among all points (nodes) in the element
            for (int i = this->pts_offset; i < this->pts_offset + this->n_pts; ++i) {
                int node_idx = elem_nodes[i]; // read the index of the element in the elem_nodes vector
                auto node = nodes[node_idx]; // read the element node (containing x,y,z coordinates) from the nodes vector

                // the centroid is the arithmetic avg of the coordinates
                Cx += node.x;
                Cy += node.y;
                Cz += node.z;
            }

            Cx /= this->n_pts;
            Cy /= this->n_pts;
            Cz /= this->n_pts;

            c = Vector3{ Cx, Cy, Cz};
           
        }
        else {
            // subdivide into two tetras, total volume is the sum of their volumes
            std::vector<Vector3> tetra1{ element_nodes[0], element_nodes[1], element_nodes[3], element_nodes[4] };
            std::vector<Vector3> tetra2{ element_nodes[1], element_nodes[3], element_nodes[2], element_nodes[4] };


            Vector3 centroid1 = tetra_centroid(tetra1);
            Vector3 centroid2 = tetra_centroid(tetra2);

            double volume1 = tetra_volume(tetra1);
            double volume2 = tetra_volume(tetra2);

            volume = 0;
            for (auto i : { volume1, volume2 }) {
                volume += i;
            }

            //Cx = (centroid1.x * volume1 + centroid2.x * volume2) / this->volume;
            //Cy = (centroid1.y * volume1 + centroid2.y * volume2) / this->volume;
            //Cz = (centroid1.z * volume1 + centroid2.z * volume2) / this->volume;

            double Cx = 0.0;
            double Cy = 0.0;
            double Cz = 0.0;

            // iterate among all points (nodes) in the element
            for (int i = this->pts_offset; i < this->pts_offset + this->n_pts; ++i) {
                int node_idx = elem_nodes[i]; // read the index of the element in the elem_nodes vector
                auto node = nodes[node_idx]; // read the element node (containing x,y,z coordinates) from the nodes vector

                // the centroid is the arithmetic avg of the coordinates
                Cx += node.x;
                Cy += node.y;
                Cz += node.z;
            }

            Cx /= this->n_pts;
            Cy /= this->n_pts;
            Cz /= this->n_pts;

            c = Vector3{ Cx, Cy, Cz };
        }

        break;
    }
    }
}

void Connection::calculate_centroid(const std::vector<Vector3>& nodes, const std::vector<index_t>& conn_nodes)
{
	// the centroid location on x,y,z direction
	double Cx = 0.0;
	double Cy = 0.0;
	double Cz = 0.0;

	// iterate among all points (nodes) in the element
	for (int i = this->pts_offset; i < this->pts_offset + this->n_pts; ++i) {
		int node_idx = conn_nodes[i]; // read the index of the element in the elem_nodes vector
		auto node = nodes[node_idx]; // read the element node (containing x,y,z coordinates) from the nodes vector

		// the centroid is the arithmetic avg of the coordinates
		Cx += node.x;
		Cy += node.y;
		Cz += node.z;
	}

	Cx /= this->n_pts;
	Cy /= this->n_pts;
	Cz /= this->n_pts;

	// set the centroid value
	this->c = Vector3{ Cx, Cy, Cz };
}

void Connection::calculate_area(const std::vector<Vector3>& nodes, const std::vector<index_t>& conn_nodes) {

	double A = 0.0;
	auto temp1 = nodes[conn_nodes[this->pts_offset]] - nodes[conn_nodes[this->pts_offset + 1]];

	if (this->n_pts == 2) {
		A += temp1.norm();
	}
	else if (this->n_pts == 3) {
		auto temp2 = nodes[conn_nodes[this->pts_offset]] - nodes[conn_nodes[this->pts_offset + 2]];
		A += 0.5 * cross(temp1, temp2).norm();
	}
	else if (this->n_pts == 4) {
		auto temp2 = nodes[conn_nodes[this->pts_offset]] - nodes[conn_nodes[this->pts_offset + 2]];
        auto temp3 = nodes[conn_nodes[this->pts_offset]] - nodes[conn_nodes[this->pts_offset + 3]];
        auto temp4 = nodes[conn_nodes[this->pts_offset + 1]] - nodes[conn_nodes[this->pts_offset + 2]];
        auto temp5 = nodes[conn_nodes[this->pts_offset + 1]] - nodes[conn_nodes[this->pts_offset + 3]];
        auto temp6 = nodes[conn_nodes[this->pts_offset + 2]] - nodes[conn_nodes[this->pts_offset + 3]];

		A += 0.5 * cross(temp1, temp3).norm();
		A += 0.5 * cross(temp3, temp6).norm();
        A += 0.5 * cross(temp6, temp4).norm();
        A += 0.5 * cross(temp4, temp1).norm();
        A /= 2;
	}

	this->area = A;
}

void Connection::calculate_normal(const std::vector<Vector3>& nodes, const std::vector<index_t>& conn_nodes, const std::vector<mesh::Elem>& elems, const std::vector<index_t>& elem_nodes) {

	// For qudrangle and triangle faces find normal vector through crossprod of face edges
	if (this->n_pts == 4 || this->n_pts == 3) 
    {
		auto temp1 = nodes[conn_nodes[this->pts_offset]] - nodes[conn_nodes[this->pts_offset + 2]];
		auto temp2 = nodes[conn_nodes[this->pts_offset]] - nodes[conn_nodes[this->pts_offset + 1]];
		auto crossprod = cross(temp1, temp2);

        Vector3 el1_node;
        const auto& el1 = elems[this->elem_id1];
        for (int i = 0; i < 3; i++) {
            if (elem_nodes[el1.pts_offset + i] != conn_nodes[this->pts_offset] && elem_nodes[el1.pts_offset + i] != conn_nodes[this->pts_offset + 1]) {
                el1_node = nodes[elem_nodes[el1.pts_offset + i]];
                break;
            }
        }

        value_t sign = 1.0;
        if (dot(crossprod, this->c - el1_node) < 0.0)
        {
            sign = -1.0;
        }
		this->n = sign * crossprod / crossprod.norm();
	}
	else if (this->n_pts == 2) 
    {
		auto temp1 = nodes[conn_nodes[this->pts_offset]] - nodes[conn_nodes[this->pts_offset + 1]];
		Vector3 el1_node, el2_node;
		// Find second node of element 1 to use for cross product
		// Compare node IDs of connection face and element 1
		// (the elements for these kinds of connections are 2D so it's enough to loop i from 0 to 3)
        const auto& el1 = elems[this->elem_id1];
        const auto& el2 = elems[this->elem_id2];

		for (int i = 0; i < 3; i++) {
 			if (elem_nodes[el1.pts_offset + i] != conn_nodes[this->pts_offset] && elem_nodes[el1.pts_offset + i] != conn_nodes[this->pts_offset + 1]) {
				el1_node = nodes[elem_nodes[el1.pts_offset + i]];
				break;
			}
		}
        for (int i = 0; i < 3; i++) {
            if (elem_nodes[el2.pts_offset + i] != conn_nodes[this->pts_offset] && elem_nodes[el2.pts_offset + i] != conn_nodes[this->pts_offset + 1]) {
                el2_node = nodes[elem_nodes[el2.pts_offset + i]];
                break;
            }
        }

		auto temp11 = nodes[conn_nodes[this->pts_offset]] - el1_node;
		auto temp_crossprod1 = cross(temp1, temp11);
		auto normal1 = cross(temp1, temp_crossprod1);
        normal1 /= normal1.norm();

        auto temp22 = el2_node - nodes[conn_nodes[this->pts_offset]];
        auto temp_crossprod2 = cross(temp1, temp22);
        auto normal2 = cross(temp1, temp_crossprod2);
        normal2 /= normal2.norm();

        auto normal = -(normal1 + normal2) / 2.0;
        value_t sign = 1.0;
        if (dot(normal, this->c - el1_node) < 0.0)
            sign = -1.0;

		this->n = sign * normal / normal.norm();
	}

}
