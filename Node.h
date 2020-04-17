#pragma once
#define _USE_MATH_DEFINES
#include <complex>
#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <execution>
#include <iostream>
#include <limits>

// Algorithm due to Piotr Kowalczyk,
// "Global Complex Roots and Poles Finding Algorithm Based on Phase Analysis
// for Propagation and Radiation Problems"
// https://arxiv.org/pdf/1806.06522.pdf
// July 25, 2018 pre-print.
//
// This implementation uses a different form of adaptive mesh than the one
// described in the paper. Instead of using Delaunay triangulation to connect
// the midpoint nodes, adding additional nodes and reforming the mesh when
// triangles get too thin, it connects the midpoint nodes and extends the
// subdivided region so that any missing edges are not adjacent to the
// candidate edges.

namespace zf
{
	// To use, just apply the function, solve(), below. The rest is
	// implementation, but one could perhaps use the Mesh object directly to
	// cache results.
	//
	// Arguments:
	//
	// ULcorner, LRcorner : Upper-left and lower-right corners of the 
	//		rectangular search region.
	// precision: Final length of mesh edges will be less than this.
	// f: Any analytic function.
	// return_0_order: If true, returns points of order 0 (discards if false).
	// initial_mesh_len: starting length of mesh edges. Smaller is less likely 
	//		to miss a zero, but slower. -1 (default) means the function will
	//		try to choose something appropriate.
	//
	// Returns:
	// vector of results for each point. first is the location of the zero/pole,
	// second is the order. If the order is zero, it may be a regular point, or
	// it may signify an equal number of zeros and poles in the same region.
	// The algorithm should catch all zeros and poles with a sufficiently fine
	// initial mesh, but it cannot know whether it caught every one of them.
	// If the it returns fewer zeros or poles than expected, try a finer mesh.

	template <typename T>
	std::vector<std::pair<std::complex<T>, int>> solve(std::complex<T> ULcorner,
		std::complex<T> LRcorner, T precision,
		std::function<std::complex<T>(std::complex<T>)> f,
		bool return_0_order = false, T initial_mesh_len = -1);


	template <typename T>
	class Mesh;
	template <typename T>
	class Edge;
	template <typename T>
	class Node;
	template <typename T>
	class Triangle;

	// Directions radiating hexagonally from a node.
	enum class Direction
	{
		right = 0,
		up_right,
		up_left,
		left,
		down_left,
		down_right,
	};

	// Left or right side of an edge
	enum class Side
	{
		left = false,
		right = true
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
		Node(cplx loc, cplx val);
		cplx location; // Geometric location of node
		int q = 0; // Quadrant of value

		Edge<T>* get_edge(int dir);
		void set_edge(Edge<T>* e, int dir);
		void calc_q(cplx z);
	private:
		Edge<T>* edges[6] = {};
	};

	// Edge for triangular mesh of complex numbers. Each edge connects two
	// nodes. Edges have one of six directions, mainly for facilitating 
	// traversal of the mesh.
	template <typename T>
	class Edge
	{
		typedef std::complex<T> cplx;
	public:
		Edge(Node<T>* node1, Node<T>* node2, int dir);
		~Edge();
		Node<T>* get_node(int index) const;
		void set_node(int index, Node<T>* node);
		Triangle<T>* get_tri(int index) const;
		void set_tri(Side index, Triangle<T>* tri);
		int get_dir();
		// Returns the clockwise-adjacent edge connected to the specified
		// get_node.
		Edge<T>* get_next_CW(int node) const;
		// Returns the counter-clockwise-adjacent edge connected to the 
		// specified get_node.
		Edge<T>* get_next_CCW(int node) const;
		// Returns the connected edge in the same direction as this attached to
		// node 1;
		Edge<T>* get_continuation();
		// Return the edge "steps" steps counter-clockwise around node;  
		Edge<T>* get_next(int node, int steps);

		// Swaps the pointers for node 0 and node 1 and reverses the edge
		// direction. The actual nodes are unaffected.
		void reverse();

		T length();

		// sets nodes to nullptr without modifying the nodes' stored
		// connections. If any temporary edges are created during intermediate
		// calculations, this must be called before destroying them.
		void detach_nodes();

		int get_dq() const;
		void calc_dq();

		bool boundary = false;
		bool is_new = false;
	private:
		Node<T>* nodes[2] = {};
		// Triangles are only needed after adapt_mesh().
		// tris[0] should be triangle on left side.
		Triangle<T>* tris[2] = {};
		int direction; // See enum Direction for meaning of values. 
		int dq = 0; // Difference in quadrant of value for this edge's nodes.
	};

	// Triangles for finding boundary regions. Not used for constructing or
	// refining the mesh. These are only created after those things are done.
	template <typename T>
	class Triangle
	{
		typedef std::complex<T> cplx;
	public:
		Triangle(Edge<T>* a, Side a_side, Edge<T>* b,
			Side b_side, Edge<T>* c, Side c_side);
		Triangle<T>* get_adjacent(int index) const;
		Edge<T>* get_edge(int index) const;
		// is any edge external
		bool is_external();
		// is the edge at this index external
		bool is_external(int index);
		void make_CCW();

		cplx get_center();

		bool visited = false;
	private:
		Edge<T>* edges[3];
	};
};
namespace std
{
	typedef std::pair<int64_t, int64_t> key;
	template <>
	class hash<key> {
	public:
		size_t operator()(const key& name) const
		{
			return hash<int64_t>()(name.first) ^ hash<int64_t>()(name.second);
		}
	};
};
namespace zf
{
	template <typename T>
	class Mesh
	{
		typedef std::complex<T> cplx;
	public:
		Mesh(cplx corner1, cplx corner2, T init_prec, T final_prec,
			std::function<cplx(cplx)> func, bool zero_order);

		Node<T>* insert_node(cplx location);

		// Creates a get_node at the midpoint of this edge. Chosen edge will run
		// from n1 to the new point, a new edge will be created from n2 to the
		// new point.
		Node<T>* split(Edge<T>* edge, T proportion = 0.5);

		// Creates an edge between two nodes, with direction specified from n1
		// to n2.
		void connect(Node<T>* n1, Node<T>* n2, int dir);

		// Looks for edges that would form triangles to the left and right of 
		// this edge. Creates them if they do not exist. Creates new nodes in 
		// temp_nodes if necessary, and searches there first to avoid
		// duplicating new nodes.
		void complete_quad(Edge<T>* e);
		// Calls complete_quad<Edge<T>*) with a temporary edge between the two
		// given nodes.
		void complete_quad(Node<T>* n0, Node<T>* n1, int dir);

		// Returns estimated zeros/poles. pair includes the location of the
		// zero/pole and the order (negative for poles, zero for a regular point
		// or a zero/pole pair).
		std::vector<std::pair<cplx, int>> find_zeros_and_poles();

		// Increases the resolution of the mesh around the candidate edges to
		// improve the accuracy of the estimation. Call repeatedly as necessary.
		void adapt_mesh();

		// Brings candidate edges to the front and returns an iterator to the
		// end of candidate edges.
		auto candidates_to_front();

		// Creates triangles bordering each candidate edge. Requires those edges
		// to exist. Reorders edges to put the candidates at the front as a
		// side effect.
		auto create_region_triangles();

		// Does not preserve order of edges. If candidate edges have incomplete
		// triangles on either side, complete them recursively (i.e 
		// complete_triangles again for newly any generated candidates). Returns
		// an iterator to the end of the candidate edges.
		void extend_mesh();

		void clear_flags();

		// Removes all edges that are not marked as split (and therefore not
		// in a candidate region). Some edges near but outside the region will
		// remain.
		void cull_edges();

		std::pair<int64_t, int64_t> gen_key(cplx z);
	private:
		std::unordered_map<std::pair<int64_t, int64_t>,
			std::unique_ptr<Node<T>>> nodes;
		std::vector<std::unique_ptr<Edge<T>>> edges;
		std::vector<std::unique_ptr<Triangle<T>>> triangles;

		std::function<cplx(cplx)> f;
		T precision;
		bool return_0_order = false;
	};

	template<typename T>
	inline Mesh<T>::Mesh(cplx corner1, cplx corner2, T edge_width, T final_prec,
		std::function<cplx(cplx)> func, bool zero_order)
		: precision(final_prec), f(func), return_0_order(zero_order)
	{
		// ensures corner1 is the top left and corner2 is bottom right.
		if (corner2.real() < corner1.real())
			std::swap(reinterpret_cast<T(&)[2]>(corner1)[0],
				reinterpret_cast<T(&)[2]>(corner2)[0]);
		if (corner2.imag() > corner1.imag())
			std::swap(reinterpret_cast<T(&)[2]>(corner1)[1],
				reinterpret_cast<T(&)[2]>(corner2)[1]);

		const T width = corner2.real() - corner1.real();
		const T height = corner1.imag() - corner2.imag();

		const int col_count = ceil(width / edge_width);

		// The mesh will be constructed as an equilateral triangular grid, so we
		// need the height of the triangles.
		const T row_width = edge_width * sin(M_PI / 3);

		const int row_count = 1 + (int)ceil(height / row_width);
		const int node_count = row_count * col_count;
		// 100 is an arbitrary bit of extra space for new points in
		// adapted mesh.
		nodes.reserve(node_count + 100);
		// Likewise, 200 is arbitrary.
		edges.reserve(3 * node_count - 2 * (row_count + col_count) + 1 + 200);

		const T half_edge_width = edge_width / 2;
		T loc_y = corner1.imag();
		T loc_x;

		// create the nodes
		for (int y = 0; y < row_count; y++)
		{
			if (y % 2) loc_x = corner1.real() - half_edge_width;
			else loc_x = corner1.real();

			for (int x = 0; x < col_count; x++)
			{
				insert_node(cplx(loc_x, loc_y));
				loc_x += edge_width;
			}

			loc_y -= row_width;
		}

		// Lambdas for readability in the following get_node connection section.

		auto connect_R = [=](T loc_x, T loc_y)
		{
			connect(nodes[gen_key(cplx(loc_x, loc_y))].get(),
				nodes[gen_key(cplx(loc_x + edge_width, loc_y))].get(),
				(int)Direction::right);
		};

		auto connect_DL = [=](T loc_x, T loc_y)
		{
			connect(nodes[gen_key(cplx(loc_x, loc_y))].get(),
				nodes[gen_key(cplx(loc_x - half_edge_width,
					loc_y - row_width))].get(),
				(int)Direction::down_left);
		};

		auto connect_DR = [=](T loc_x, T loc_y)
		{
			connect(nodes[gen_key(cplx(loc_x, loc_y))].get(),
				nodes[gen_key(cplx(loc_x + half_edge_width,
					loc_y - row_width))].get(),
				(int)Direction::down_right);
		};

		// Nodes are connected in an equilateral triangular grid, with the bases
		// aligned with the horizontal axis.

		// TODO: Simplify this, now that structure has changed.

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
		loc_y = corner1.imag();
		for (y = 0; y < row_count - 1; y++)
		{
			x = 0;
			if (y % 2) loc_x = corner1.real() - half_edge_width;
			else loc_x = corner1.real();
			if (y % 2)
			{
				//  ~~~~~
				//   ~
				//    ~ 

				connect_R(loc_x, loc_y);
				connect_DR(loc_x, loc_y);
				x++;
				loc_x += edge_width;
				for (x; x < col_count - 1; x++)
				{
					//   +++++
					//  + +
					// +   + 

					connect_R(loc_x, loc_y);
					connect_DL(loc_x, loc_y);
					connect_DR(loc_x, loc_y);
					loc_x += edge_width;
				}

				//   #
				//  # #
				// #   #

				connect_DR(loc_x, loc_y);
				connect_DL(loc_x, loc_y);
			}
			else
			{
				for (x; x < col_count - 1; x++)
				{
					//   -----
					//  / \
					// /   \ 

					connect_R(loc_x, loc_y);
					connect_DL(loc_x, loc_y);
					connect_DR(loc_x, loc_y);
					loc_x += edge_width;
				}
				//
				//  $
				// $

				connect_DL(loc_x, loc_y);
			}
			loc_y -= row_width;
		}
		if (y % 2) loc_x = corner1.real() - half_edge_width;
		else loc_x = corner1.real();
		for (x = 0; x < col_count - 1; x++)
		{
			//
			// *****
			//
			connect_R(loc_x, loc_y);
			loc_x += edge_width;
		}

		extend_mesh();
	}

	template<typename T>
	inline Node<T>* Mesh<T>::insert_node(cplx location)
	{
		return nodes.insert({ gen_key(location), std::make_unique<Node<T>>(
			location, f(location)) }).first->second.get();
	}

	template<typename T>
	inline Node<T>* Mesh<T>::split(Edge<T>* edge, T t)
	{
		 auto N = insert_node(t * edge->get_node(0)->location + (1 - t) *
			edge->get_node(1)->location);
		connect(edge->get_node(1), N, edge->get_dir() + 3);
		edge->set_node(1, N);
		edge->is_new = true;
		edges.back()->is_new = true;
		return N;
	}

	template<typename T>
	inline void Mesh<T>::connect(Node<T>* n1, Node<T>* n2, int dir)
	{
		edges.push_back(std::make_unique<Edge<T>>(n1, n2, mod(dir, 6)));
		n1->set_edge(edges.back().get(), dir);
		n2->set_edge(edges.back().get(), dir + 3);
	}

	template<typename T>
	inline void Mesh<T>::complete_quad(Edge<T>* e)
	{
		auto add_edges = [&](bool right) {
			int n0 = 0, n1 = 1;
			int dir_change1 = 1, dir_change2 = 2;
			if (!right)
			{
				std::swap(n0, n1);
				std::swap(dir_change1, dir_change2);
			}
			auto neg = n0 - n1;

			// Can't have a candidate edge on the edge of the mesh, so we
			// apply the same process to extend the newest edge if it is one.
			auto recurse_if_candidate = [&](Edge<T>* e)
			{
				if (abs(e->get_dq()) == 2)
				{
					complete_quad(e);
				}
			};

			// If one edge does not exist, uses the second point from the other
			// edge to create it.
			auto add_one_edge = [&](bool CCW) {
				int dir = CCW ? 1 : -1;
				int dir_change = CCW ? dir_change1 : dir_change2;
				int n_0 = n0;
				int n_1 = n1;
				if (!CCW) std::swap(n_0, n_1);

				// N needs to be the node not shared with e, so we store one,
				// check it, and if matches, store the other, instead.
				auto e2 = e->get_next(n_1, dir);
				auto N = e2->get_node(n_1);
				if (N == e->get_node(n_1))
					N = e2->get_node(n_0);
				auto elen = e->length();
				auto e2len = e2->length();

				if (!e2->is_new)
				{
					connect(e->get_node(n_0), N, e->get_dir()
						+ neg * dir_change);
					recurse_if_candidate(edges.back().get());
				}
				else
				{
					e2 = e2->get_continuation();
					connect(e->get_node(n_0), e2->get_node(0),
						e->get_dir() + neg * dir_change);
					recurse_if_candidate(edges.back().get());
				}
			};

			// If neither edge of the triangle on the target side exists, then
			// add a node (or find one in temp_nodes) and create the edges.
			if (!e->get_next_CCW(n1) && !e->get_next_CW(n0))
			{
				// places the node by taking e's node 0's location and
				// adding a number with the same length as e, rotated
				// appropriately.
				auto N = insert_node(e->get_node(n0)->location + e->length()
					* exp((e->get_dir() + dir_change1 * neg)
						* M_PI / 3 * cplx(0, 1)));

				connect(e->get_node(n0), N, e->get_dir() + neg * dir_change1);
				recurse_if_candidate(edges.back().get());
				connect(e->get_node(n1), N, e->get_dir() + neg * dir_change2);
				recurse_if_candidate(edges.back().get());
			}
			else if (!e->get_next_CW(n0))
			{
				add_one_edge(true);
			}
			else if (!e->get_next_CCW(n1))
			{
				add_one_edge(false);
			}
		};

		add_edges(true);
		add_edges(false);
	}

	template<typename T>
	inline void Mesh<T>::complete_quad(Node<T>* n0, Node<T>* n1, int dir)
	{
		Edge<T> e(n0, n1, dir);
		complete_quad(&e);
		e.detach_nodes();
	}

	template<typename T>
	inline std::vector<std::pair<std::complex<T>, int>>
		Mesh<T>::find_zeros_and_poles()
	{
		auto candidates_end = create_region_triangles();

		// Finds the order for each candidate region by looking at a constituent
		// triangle, adding up the dqs for each external edge, then recursively
		// checking each adjacent triangle on the internal edges. The function
		// estimates the location of a zero/pole by averaging the boundary
		// points and the order by adding up the dq values for each edge
		// (4 quadrants = 1 full turn).

		std::vector<std::pair<cplx, int>> points;

		std::for_each(triangles.begin(), triangles.end(), [&](auto&& triangle)
			{
				// Center of triangle is just a convenient point 
				// guaranteed to be inside its region.
				cplx interior_point = triangle->get_center();
				int boundarysize = 0;
				float sum_dq = 0;
				cplx node_sum = 0;
				Edge<T>* next;
				std::function<void(Triangle<T>*)> next_tri =
					[&](Triangle<T>* tri)
				{
					if (tri && !tri->visited)
					{
						tri->visited = true;
						tri->make_CCW();
						for (int i = 0; i < 3; i++)
						{
							if (tri->is_external(i))
							{
								boundarysize++;
								next = tri->get_edge(i);
								node_sum += next->get_node(0)->location
									+ next->get_node(1)->location;
								sum_dq += next->get_dq();
							}
							else
								next_tri(tri->get_adjacent(i));
						}
					}
				};
				next_tri(triangle.get());

				if (boundarysize)
				{
					points.push_back(std::make_pair(node_sum
						/ (cplx)(boundarysize * 2), sum_dq / 4));
				}
			});

		return points;
	}

	template<typename T>
	inline void Mesh<T>::adapt_mesh()
	{
		auto candidate_end = candidates_to_front();

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
		std::function<void(Edge<T>*)> subdivide = [&](Edge<T>* e) {
			auto e2 = e->get_continuation();

			auto split_and_get_end_node = [&](Edge<T>* E)
			{
				if (!E->is_new)
				{
					split(E);
				}
				return E->get_node(1);
			};
			
			complete_quad(e->get_node(0), e2->get_node(0), e->get_dir());

			int dir = e->get_dir();

			std::pair<Node<T>*, int> new_nodes[4] = {
			{ split_and_get_end_node(e->get_next_CW(0)), (dir - 2) },
			{ split_and_get_end_node(e2->get_next_CCW(0)), (dir - 1) },
			{ split_and_get_end_node(e2->get_next_CW(0)), (dir + 1) },
			{ split_and_get_end_node(e->get_next_CCW(0)), (dir + 2) },
			};

			Edge<T>* new_edges[8] = {
				e->get_next_CW(0), e->get_next_CW(0)->get_continuation(),
				e2->get_next_CCW(0), e2->get_next_CCW(0)->get_continuation(),
				e2->get_next_CW(0), e2->get_next_CW(0)->get_continuation(),
				e->get_next_CCW(0), e->get_next_CCW(0)->get_continuation()
			};
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
				if (!candidate_midpoint->get_edge(N.second))
				{
					connect(candidate_midpoint, N.first, N.second);
					edges.back()->is_new = true;
				}
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
			if (!new_nodes[0].first->get_edge(dir))
			{
				connect(new_nodes[0].first, new_nodes[1].first, dir);
				edges.back()->is_new = true;
			}
			if (!new_nodes[3].first->get_edge(dir))
			{
				connect(new_nodes[3].first, new_nodes[2].first, dir);
				edges.back()->is_new = true;
			}
			for (auto ne : new_edges)
			{
				if (!ne->boundary && abs(ne->get_dq()) == 2)
				{
					ne->boundary = true;
					subdivide(ne);
				}
			}
		};

		for (size_t i = 0; i < dist; i++)
		{
			subdivide(edges[i].get());
		}
	}

	template<typename T>
	inline auto Mesh<T>::candidates_to_front()
	{
		return std::partition(edges.begin(), edges.end(),
			[](auto&& e)
			{
				return abs(e->get_dq()) == 2;
			});
	}

	template<typename T>
	inline auto Mesh<T>::create_region_triangles()
	{
		auto candidate_end = candidates_to_front();

		// For each candidate edge, create the triangles bordering it, if
		// it doesn't already exist.

		std::for_each(edges.begin(), candidate_end, [&](auto&& e)
			{
				e->boundary = true;
				auto e1 = e->get_next_CW(0);
				Side e1_side;
				if (e1->get_node(0) == e->get_node(0))
					e1_side = Side::left;
				else
					e1_side = Side::right;

				Side e2_side;
				auto e2 = e->get_next_CCW(1);
				if (e2->get_node(1) == e->get_node(1))
					e2_side = Side::left;
				else
					e2_side = Side::right;

				Side e3_side;
				auto e3 = e->get_next_CW(1);
				if (e3->get_node(1) == e->get_node(1))
					e3_side = Side::right;
				else
					e3_side = Side::left;

				Side e4_side;
				auto e4 = e->get_next_CCW(0);
				if (e4->get_node(0) == e->get_node(0))
					e4_side = Side::right;
				else
					e4_side = Side::left;
				if (!e->get_tri(1))
					triangles.push_back(std::make_unique<Triangle<T>>(
						e.get(), Side::right, e1, e1_side, e2, e2_side));
				if (!e->get_tri(0))
					triangles.push_back(std::make_unique<Triangle<T>>(
						e.get(), Side::left, e3, e3_side, e4, e4_side));
			});

		return candidate_end;
	}

	template<typename T>
	inline void Mesh<T>::extend_mesh()
	{
		auto C = candidates_to_front();
		auto dist = std::distance(edges.begin(), C);
		for (int i = 0; i < dist; i++)
		{
			auto e = edges[i].get();
			if (!(e->get_next_CW(0) && e->get_next_CCW(1) &&
				e->get_next_CCW(0) && e->get_next_CW(1))) complete_quad(e);
		};
	}

	template<typename T>
	inline void Mesh<T>::clear_flags()
	{
		std::for_each(/*std::execution::par_unseq,*/ edges.begin(), edges.end(),
			[](auto&& e)
			{
				e->boundary = false;
				e->is_new = false;
			});
	}

	template<typename T>
	inline void Mesh<T>::cull_edges()
	{
		edges.erase(std::remove_if(edges.begin(), edges.end(), [](auto&& e)
			{
				return !e->is_new;
			}), edges.end());
	}

	template<typename T>
	inline std::pair<int64_t, int64_t> Mesh<T>::gen_key(cplx z)
	{
		int64_t a = z.real() / precision * 100 + 0.5;
		int64_t b = z.imag() / precision * 100 + 0.5;
		return std::make_pair(a, b);
	}

	template<typename T>
	inline Edge<T>::Edge(Node<T>* node1, Node<T>* node2, int dir)
		: nodes{ node1, node2 }, direction(dir)
	{
		calc_dq();
	}

	template<typename T>
	inline Edge<T>::~Edge()
	{
		if (nodes[0]) nodes[0]->set_edge(nullptr, direction);
		if (nodes[1]) nodes[1]->set_edge(nullptr, direction + 3);
	}

	template<typename T>
	inline Node<T>* Edge<T>::get_node(int index) const
	{
		return nodes[index];
	}

	template<typename T>
	inline void Edge<T>::set_node(int i, Node<T>* N)
	{
		nodes[i] = N;
		N->set_edge(this, direction + i * 3);
		calc_dq();
	}

	template<typename T>
	inline Triangle<T>* Edge<T>::get_tri(int index) const
	{
		return tris[index];
	}

	template<typename T>
	inline void Edge<T>::set_tri(Side index, Triangle<T>* tri)
	{
		tris[(int)index] = tri;
	}

	template<typename T>
	inline int Edge<T>::get_dir()
	{
		return direction;
	}

	template<typename T>
	inline Edge<T>* Edge<T>::get_next_CW(int n) const
	{
		return this->get_node(n)->get_edge(this->direction - 1 + 3 * n);
	}

	template<typename T>
	inline Edge<T>* Edge<T>::get_next_CCW(int n) const
	{
		return this->get_node(n)->get_edge(this->direction + 1 + 3 * n);
	}

	template<typename T>
	inline Edge<T>* Edge<T>::get_continuation()
	{
		return nodes[1]->get_edge(direction);
	}

	template<typename T>
	inline Edge<T>* Edge<T>::get_next(int n, int steps)
	{
		return this->get_node(n)->get_edge(this->direction + steps + 3 * n);
	}

	template<typename T>
	inline void Edge<T>::reverse()
	{
		std::swap(nodes[0], nodes[1]);
		std::swap(tris[0], tris[1]);
		direction += 3;
		direction %= 6;
		dq *= -1;
	}

	template<typename T>
	inline T Edge<T>::length()
	{
		return abs(nodes[1]->location - nodes[0]->location);
	}

	template<typename T>
	inline void Edge<T>::detach_nodes()
	{
		nodes[0] = nullptr;
		nodes[1] = nullptr;
	}

	template<typename T>
	inline int Edge<T>::get_dq() const
	{
		return dq;
	}

	template<typename T>
	inline void Edge<T>::calc_dq()
	{
		dq = (nodes[1]->q - nodes[0]->q);
		if (dq > 2) dq -= 4;
		else if (dq < -2) dq += 4;
	}

	template<typename T>
	inline std::vector<std::pair<std::complex<T>, int>>
		solve(std::complex<T> ULcorner, std::complex<T> LRcorner, T precision,
			std::function<std::complex<T>(std::complex<T>)> f,
			bool return_0_order, T initial_mesh_len)
	{
		// Arbitrary. Later will add a trial-and error mesh sizer which tests
		// a few values in the desired neighborhood and picks the one which
		// finds the most candidate regions.
		if (initial_mesh_len < 0)
			initial_mesh_len = std::min(abs(ULcorner.real() - LRcorner.real())
				/ 20.1, abs(ULcorner.imag() - LRcorner.imag()) / 20.1);
		int iterations = std::log2(initial_mesh_len / precision);
		Mesh<double> mesh(ULcorner, LRcorner, initial_mesh_len, precision, f,
			return_0_order);

		for (int i = 0; i < iterations; i++)
		{
			mesh.adapt_mesh();
			mesh.cull_edges();
			mesh.clear_flags();
		}
		return mesh.find_zeros_and_poles();
	}
	template<typename T>
	inline Triangle<T>::Triangle(Edge<T>* a, Side a_side, Edge<T>* b,
		Side b_side, Edge<T>* c, Side c_side)
		: edges{ a, b, c }
	{
		a->set_tri(a_side, this);
		b->set_tri(b_side, this);
		c->set_tri(c_side, this);
	}
	template<typename T>
	inline Triangle<T>* Triangle<T>::get_adjacent(int index) const
	{
		auto edge = edges[index];
		if (this == edge->get_tri(0))
			return edge->get_tri(1);
		else return edge->get_tri(0);
	}
	template<typename T>
	inline Edge<T>* Triangle<T>::get_edge(int index) const
	{
		return edges[index];
	}
	template<typename T>
	inline bool Triangle<T>::is_external()
	{
		return get_adjacent(0) || get_adjacent(1) || get_adjacent(2);
	}
	template<typename T>
	inline bool Triangle<T>::is_external(int index)
	{
		return !get_adjacent(index);
	}
	template<typename T>
	inline void Triangle<T>::make_CCW()
	{
		for (auto e : edges)
		{
			if (e->get_tri(0) != this) e->reverse();
		}
	}
	template<typename T>
	inline std::complex<T> Triangle<T>::get_center()
	{
		cplx res = 0;
		for (auto e : edges)
			res += e->get_node(0)->location;
		return res / 3.0;
	}
	template<typename T>
	inline Node<T>::Node(cplx loc, cplx val)
		: location(loc)
	{
		calc_q(val);
	}
	template<typename T>
	inline Edge<T>* Node<T>::get_edge(int dir)
	{
		return edges[mod(dir, 6)];
	}
	template<typename T>
	inline void Node<T>::set_edge(Edge<T>* e, int dir)
	{
		edges[mod(dir, 6)] = e;
	}
	template<typename T>
	inline void Node<T>::calc_q(cplx z)
	{
		if (T a = std::arg(z); a >= 0) q = ceil(a / M_PI_2);
		else q = 4 + ceil(a / M_PI_2);
	}
}
