#pragma once
#define _USE_MATH_DEFINES
#include <complex>
#include <vector>
#include <array>
#include <functional>
#include <algorithm>
#include <execution>
#include <iostream>

template <typename T>
class Edge;
template <typename T>
class Node;

enum Direction
{
    right = 0,
    up_right,
    up_left,
    left,
    down_left,
    down_right,
};

// c++'s built-in mod operator actually gives division-remainder.
inline int mod(int a, int b)
{
	a %= b;
	if (a >= 0) return a;
	else return b + a;
}

// Node for a trianglular mesh of complex numbers. Each node can have up to six
// connections to other nodes, represented by Edges.
template <typename T>
class Node
{
	typedef std::complex<T> cplx;
public:
	Node(cplx loc) : location(loc) {}
	cplx location; // Geometric location of node
	cplx value; // Result of some function applied to node. Should optimize it out.
	int q = 0; // Quadrant of value

	Edge<T>* get_edge(int dir) { return edges[mod(dir, 6)]; }
	void set_edge(Edge<T>* e, int dir) { edges[mod(dir, 6)] = e; }
	void calc_q()
	{
		if (auto a = arg(value); a >= 0) q = ceil(a / M_PI_2);
		else q = 4 + ceil(a / M_PI_2);
	}

private:
	Edge<T>* edges[6] = {};
};

// Edge for triangular mesh of complex numbers. Each edge connects two nodes.
// Edges have one of six directions, mainly for facilitating traversal of the
// mesh.
template <typename T>
class Edge
{
	typedef std::complex<T> cplx;
public:
	Edge(Node<T>* node1, Node<T>* node2, int dir);
	Node<T>* get_node(int index) const { return nodes[index]; }
	void set_node(int index, Node<T>* get_node);
	int get_dir() { return direction; }
	// Returns the clockwise-adjacent edge connected to the specified get_node.
	Edge<T>* get_next_CW(int get_node) const;
	// Returns the counter-clockwise-adjacent edge connected to the specified
	//get_node.
	Edge<T>* get_next_CCW(int get_node) const;
	// Returns the connected edge in the same direction as this.
	Edge<T>* get_continuation();
	// Returns a boundary edge attached to node 1 (ideally only one should
	// exist, but it doesn't check that for now). If the point is node 1 on
	// the found edge, swaps those nodes so the boundary has a consistent
	// orientation.
	Edge<T>* next_boundary();
	// Swaps the pointers for node 0 and node 1 and reverses the edge direction.
	// The actual nodes are unaffected.
	void reverse();

	int get_dq() const { return dq; }
	void calc_dq() { dq = (nodes[1]->q - nodes[0]->q); }

	//template <typename T>
	//friend bool operator==(const Edge<T>& e1, const Edge<T>& e2);

	unsigned char visit_count = 0, visit_left = 0, visit_right = 0;

	unsigned char boundary = 0;
	bool split = false;
private:
	Node<T>* nodes[2] = {};
	int direction; // See enum Direction for meaning of values. 
	int dq = 0; // Difference in quadrant of value for this edge's nodes.
};

template <typename T>
class Mesh
{
	typedef std::complex<T> cplx;
public:
	Mesh(cplx corner1, cplx corner2, T init_prec, T final_prec, std::function<cplx(cplx)> func);
	void add_node(cplx location);

	// Creates a get_node at the midpoint of this edge. Chosen edge will run from n1 to
	// the new point, a new edge will be created from n2 to the new point.
	// Does not automatically update dq for either edge.
	void split(Edge<T>* edge);
	// Creates an edge between two nodes, with direction specified from n1 to n2.
	void connect(Node<T>* n1, Node<T>* n2, int dir);
	
	// applies f to all nodes and calculates dq for all edges.
	void apply_all();

	// Returns estimated zeros/poles. pair includes the location of the
	// zero/pole and the order (negative for poles, zero for a regular point
	// or a zero/pole pair).
	std::vector<std::pair<cplx, int>> find_zeros_and_poles();

	// Increases the resolution of the mesh around the candidate edges to improve
	// the accuracy of the estimation, and discards points outside those regions.
	// Can be called repeatedly.
	void adapt_mesh();

	// Reorders edges to put the candidates at the front, and marks the edges of
	// the triangles they belong to (visited_left and visted_right). returns
	// iterator to the first non-candidate edge.
	auto mark_candidate_region_edges();
private:
	// The order of edges and nodes is used in constructing the mesh,
	// but it will not be preserved afterward. Generally, traversing the mesh
	// will be done with the pointers in each mesh and node via member
	// functions.
	std::vector<std::unique_ptr<Node<T>>> nodes;
	std::vector<std::unique_ptr<Edge<T>>> edges;
	std::function<cplx(cplx)> f;
	T precision;
};

template<typename T>
inline Mesh<T>::Mesh(cplx corner1, cplx corner2, T edge_width, T final_prec,
	std::function<cplx(cplx)> func)
	: precision(final_prec), f(func)
{
	// ensures corner1 is the lower left one, and corner2 is upper right.
	if (corner2.real() < corner1.real())
	std::swap(reinterpret_cast<T(&)[2]>(corner1)[0],
		reinterpret_cast<T(&)[2]>(corner2)[0]);
	if (corner2.imag() < corner1.imag())
	std::swap(reinterpret_cast<T(&)[2]>(corner1)[1],
		reinterpret_cast<T(&)[2]>(corner2)[1]);

	T width = corner2.real() - corner1.real();
	T height = corner2.imag() - corner1.imag();

	// If the user's edge width doesn't fit exactly, shrink it so that it does.
	int col_count = 1 + (int)ceil(width / edge_width);
	edge_width = width / col_count;

	// The mesh will be constructed as an equilateral triangular grid, so we
	// need the height of the triangles.
	T row_width = edge_width * sin(M_PI / 3);

	int row_count = 1 + (int)ceil(height / row_width);
	int node_count = row_count * col_count;
	// 100 is an arbitrary bit of extra space for new points in adapted mesh.
	nodes.reserve(node_count + 100);
	// Likewise, 200 is arbitrary.
	edges.reserve(3 * node_count - 2 * (row_count + col_count) + 1 + 200);

	T half_edge_width = edge_width / 2;
	T loc_y = corner1.imag();
	T loc_x;

	// create the nodes
	for (int y = 0; y < row_count; y++)
	{
		if (y % 2) loc_x = corner1.imag() - half_edge_width;
		else loc_x = corner1.imag();

		for (int x = 0; x < col_count; x++)
		{
			add_node(cplx(loc_x, loc_y));
			loc_x += edge_width;
		}

		loc_y += row_width;
	}

	// Lambdas for readability in the following get_node connection section.

	auto connect_R = [=](int x, int y)
	{
		connect(nodes[y * col_count + x].get(),
			nodes[y * col_count + x + 1].get(), Direction::right);
	};

	auto connect_DL_even = [=](int x, int y)
	{
		connect(nodes[y * col_count + x].get(),
			nodes[(y + 1) * col_count + x].get(), Direction::down_left);
	};

	auto connect_DL_odd = [=](int x, int y)
	{
		connect(nodes[y * col_count + x].get(),
			nodes[(y + 1) * col_count + x - 1].get(), Direction::down_left);
	};

	auto connect_DR_even = [=](int x, int y)
	{
		connect(nodes[y * col_count + x].get(),
			nodes[(y + 1) * col_count + x + 1].get(), Direction::down_right);
	};

	auto connect_DR_odd = [=](int x, int y)
	{
		connect(nodes[y * col_count + x].get(),
			nodes[(y + 1) * col_count + x].get(), Direction::down_right);
	};

	// Nodes are connected in an equilateral triangular grid, with the bases
	// aligned with the horizontal axis.

	//  <----------->   Repeat as necessary
	//   ----- ----- $ 
	//  / \   / \   $   <--- Even rows
	// /   \ /   \ $
	// ~~~~~ +++++ #
	//  ~   + +   # #   <--- Odd rows
	//   ~ +   + #   #
	//     ***** *****  <--- Last row
	//      <->   Repeat as necessary
	int x, y;
	for (y = 0; y < row_count -	1; y++)
	{
		x = 0;
		if (y % 2)
		{
			//  ~~~~~
			//   ~
			//    ~ 

			connect_R(x, y);
			connect_DR_odd(x, y);
			x++;
			for (x; x < col_count - 1; x++)
			{
				//   +++++
				//  + +
				// +   + 

				connect_R(x, y);
				connect_DL_odd(x, y);
				connect_DR_odd(x, y);
			}

			//   #
			//  # #
			// #   #

			connect_DR_odd(x, y);
			connect_DL_odd(x, y);
		}
		else
		{
			for (x; x < col_count - 1; x++)
			{
				//   -----
				//  / \
				// /   \ 

				connect_R(x, y);
				connect_DL_even(x, y);
				connect_DR_even(x, y);
			}
			//
			//  $
			// $

			connect_DL_even(x, y);
		}
	}
	for (x = 0; x < col_count - 1; x++)
		//
		// *****
		//
		connect_R(x, y);
	apply_all();
}

template<typename T>
inline void Mesh<T>::add_node(cplx location)
{
	nodes.push_back(std::make_unique<Node<T>>(location));
}

template<typename T>
inline void Mesh<T>::split(Edge<T>* edge)
{
	add_node((edge->get_node(0)->location + edge->get_node(1)->location) / 2.0);
	connect(edge->get_node(1), nodes.back().get(), edge->get_dir() + 3);
	edge->set_node(1, nodes.back().get());
	edge->split = true;
	edges.back()->split = true;
	edges.back()->visit_count = edge->visit_count;
	edges.back()->visit_left = edge->visit_left;
	edges.back()->visit_right = edge->visit_right;
}

template<typename T>
inline void Mesh<T>::connect(Node<T>* n1, Node<T>* n2, int dir)
{
	edges.push_back(std::make_unique<Edge<T>>(n1, n2, mod(dir, 6)));
	n1->set_edge(edges.back().get(), dir);
	n2->set_edge(edges.back().get(), dir + 3);
}

template<typename T>
inline void Mesh<T>::apply_all()
{
	std::for_each(std::execution::par_unseq, nodes.begin(), nodes.end(),
		[&](auto& N)
		{
			N->value = f(N->location);
			N->calc_q();
		});
	std::for_each(std::execution::par_unseq, edges.begin(), edges.end(),
		[](auto& E)
		{
			E->calc_dq();
		});
}

template<typename T>
inline std::vector<std::pair<std::complex<T>, int>> Mesh<T>::find_zeros_and_poles()
{
	// Boundary edges have been visited exactly once (calculated in adapt_mesh()).
	// All the others must be removed.
	auto boundary_end = std::remove_if(edges.begin(), edges.end(),
		[](auto&& e)
		{
			return e->visit_count != 1;
		});
	edges.erase(boundary_end, edges.end());

	// Follows each edge around its loop, marking each edge traversed. When it
	// cannot find a non-visited edge, the loop should be complete. This test
	// version of the loop estimates the location of the zero/pole by averaging
	// the boundary points.

	std::vector<std::pair<cplx, int>> points;
	std::for_each(edges.begin(), edges.end(), [&](auto&& e)
		{
			auto current = e.get();
			int boundarysize = 0;
			if (!(current->boundary || abs(current->get_dq()) > 1))
			{
				Edge<T>* next = nullptr;
				cplx node_sum = 0;
				int node_count = 0;
				do
				{
					next = current->next_boundary();
					if (next)
					{
						boundarysize++;
						node_sum += next->get_node(0)->location
							+ next->get_node(1)->location;
						node_count += 2;
						current = next;
					}
				} while (next);
				if (node_count)
				{
					points.push_back(
						std::make_pair(node_sum / (cplx)node_count, 0));
					std::cout << "boundary size: " << boundarysize << "\n";
				}
			}
		});
	for (auto&& P : points)
		std::cout << P.first << "\n";
	return points;
}

template<typename T>
inline void Mesh<T>::adapt_mesh()
{	
	auto candidate_end = mark_candidate_region_edges();

	auto dist = std::distance(edges.begin(), candidate_end);

	// Splits the candidate edges in half.
	for (size_t i = 0; i < dist; i++)
	{
		split(edges[i].get());
	}

		// Splits all the visited edges in half, and connect the midpoints.
		//
		//    /\
		//   /__\
		//  /\  /\
		// /__\/__\
		// \  /\  /
		//  \/__\/
		//   \  /
		//    \/

	for (size_t i = 0; i < dist; i++)
	{
		auto e = edges[i].get();
		auto e2 = e->get_continuation();

		auto split_and_get_end_node = [&](Edge<T>* E)
		{
			if (!E->split)
			{
				split(E);
			}
			return E->get_node(1);
		};

		int dir = e->get_dir();
		std::array<std::pair<Node<T>*, int>, 4> new_nodes;

		new_nodes[0] = { split_and_get_end_node(e->get_next_CW(0)), (dir - 2) };
		new_nodes[1] = { split_and_get_end_node(e2->get_next_CCW(0)), (dir - 1) };
		new_nodes[2] = { split_and_get_end_node(e2->get_next_CW(0)), (dir + 1) };
		new_nodes[3] = { split_and_get_end_node(e->get_next_CCW(0)), (dir + 2) };

		auto candidate_midpoint = e->get_node(1);

		//     .
		//    . .
		//   \   /
		//  . \ / .
		// ....X....
		//  . / \ .
		//   /   \
		//    . .
		//     .
		for (auto& N : new_nodes)
		{
			connect(candidate_midpoint, N.first, N.second);
		}
		//     .
		//    . .
		//   .___.
		//  . . . .
		// .........
		//  . . . .
		//   .___.
		//    . .
		//     .
		connect(new_nodes[0].first, new_nodes[1].first, dir);
		connect(new_nodes[3].first, new_nodes[2].first, dir);
	}

	// Removes edges which have not been visited, and are therefore outside the
	// candidate regions. Also updates visit_count.

	auto visited_end = std::remove_if(edges.begin(), edges.end(),
		[](auto&& e)
		{
			e->visit_count += e->visit_left + e->visit_right;
			return e->visit_count == 0;
		});
	edges.erase(visited_end, edges.end());
}

template<typename T>
inline auto Mesh<T>::mark_candidate_region_edges()
{
	// Brings candidate edges to the front and marks them as visited twice,
	// ensuring they will be excluded from the boundary edges.
	auto candidate_end = std::partition(edges.begin(), edges.end(),
		[](auto&& e)
		{
			if (auto x = abs(e->get_dq()); x == 2)
			{
				e->visit_count = 2;
				return true;
			}
			else return false;
		});

	// For each candidate edge, checks each edge belonging to a triangle
	// with the candidate (4 total). If one of those edges has been seen
	// both by a candidate to its left and a candidate to its right, it is
	// an interior edge and will be removed later. If it has been seen twice
	// from the same side and not from the other, it is still external.

	// TODO: handle nullptr edges. That's a sign the mesh needs to be extended,
	// So do that.

	std::for_each(edges.begin(), candidate_end, [&](auto&& e)
		{
			auto e1 = e->get_next_CW(0);
			if (e1->get_node(0) == e->get_node(0))
				e1->visit_left = 1;
			else
				e1->visit_right = 1;

			e1 = e->get_next_CCW(1);
			if (e1->get_node(1) == e->get_node(1))
				e1->visit_left = 1;
			else
				e1->visit_right = 1;

			e1 = e->get_next_CW(1);
			if (e1->get_node(1) == e->get_node(1))
				e1->visit_right = 1;
			else
				e1->visit_left = 1;

			e1 = e->get_next_CCW(0);
			if (e1->get_node(0) == e->get_node(0))
				e1->visit_right = 1;
			else
				e1->visit_left = 1;
		});
	return candidate_end;
}

template<typename T>
inline Edge<T>::Edge(Node<T>* node1, Node<T>* node2, int dir)
	: nodes{ node1, node2 }, direction(dir)
{
}

template<typename T>
inline void Edge<T>::set_node(int i, Node<T>* N)
{
	nodes[i] = N;
	N->set_edge(this, direction + i * 3);
}

template<typename T>
inline Edge<T>* Edge<T>::get_next_CW(int n) const
{
	return this->get_node(n)->get_edge(this->direction - 1 + 3*n);
}

template<typename T>
inline Edge<T>* Edge<T>::get_next_CCW(int n) const
{
	return this->get_node(n)->get_edge(this->direction + 1 + 3*n);
}

template<typename T>
inline Edge<T>* Edge<T>::get_continuation()
{
	return nodes[1]->get_edge(direction);
}

template<typename T>
inline Edge<T>* Edge<T>::next_boundary()
{
	auto N = this->get_node(1);
	Edge<T>* bdry = nullptr;
	for (int i = 0; i < 6; i++)
	{
		if (auto e = N->get_edge(i);
			e && e->visit_count == 1 && e->boundary == false)
		{
			e->boundary = true;
			bdry = e;
			break;
		}
	}
	if (bdry && bdry->get_node(1) == N)
		bdry->reverse();
	return bdry;
}

template<typename T>
inline void Edge<T>::reverse()
{
	std::swap(nodes[0], nodes[1]);
	direction += 3;
	direction %= 6;
}

//template<typename T>
//inline bool operator==(const Edge<T>& e1, const Edge<T>& e2)
//{
//	return e1.get_node(0) == e2.get_node(0) && e1.get_node(1) == e2.get_node(1);
//}
