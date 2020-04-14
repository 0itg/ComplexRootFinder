#pragma once
#define _USE_MATH_DEFINES
#include <complex>
#include <vector>
#include <array>
#include <functional>
#include <algorithm>
#include <execution>
#include <iostream>

// Algorithm due to Piotr Kowalczyk,
// "Global Complex Roots and Poles Finding Algorithm Based on Phase Analysis
// for Propagation and Radiation Problems"
// https://arxiv.org/pdf/1806.06522.pdf
// July 25, 2018 pre-print.
//
// This implementation uses a different form of adaptive mesh than the one
// described in the paper. Instead of using Delaunay triangulation to connect
// the midpoint nodes, adding additional nodes and reforming the mesh when
// triangles get too thin, it simply connects the midpoint nodes and extends
// the subdivided region so that any missing edges are not adjacent to the
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
	// initial_mesh_len: starting length of mesh edges. Smaller is less likely 
	//		to miss a zero, but slower. -1 (default) means the function will
	//		try to choose something appropriate.

	template <typename T>
	std::vector<std::pair<std::complex<T>, int>> solve(std::complex<T> ULcorner,
		std::complex<T> LRcorner, T precision,
		std::function<std::complex<T>(std::complex<T>)> f,
		T initial_mesh_len = -1);



	template <typename T>
	class Mesh;
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

	template <typename T> T angle(std::complex<T> a, std::complex<T> b)
	{
		return std::asin((a.real() * b.imag() - a.imag() * b.real()) / abs(a * b));
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
		int q = 0; // Quadrant of value

		Edge<T>* get_edge(int dir) { return edges[mod(dir, 6)]; }
		void set_edge(Edge<T>* e, int dir) { edges[mod(dir, 6)] = e; }
		void calc_q(cplx z)
		{
			if (T a = std::arg(z); a >= 0) q = ceil(a / M_PI_2);
			else q = 4 + ceil(a / M_PI_2);
		}

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
		Node<T>* get_node(int index) const { return nodes[index]; }
		void set_node(int index, Node<T>* node);
		int get_dir() { return direction; }
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

		T length() { return abs(nodes[1]->location - nodes[0]->location); }

		int get_dq() const { return dq; }
		void calc_dq();

		bool visit_left = false;
		bool visit_right = false;
		bool boundary = false;
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
		Mesh(cplx corner1, cplx corner2, T init_prec, T final_prec,
			std::function<cplx(cplx)> func);

		void add_node(cplx location);
		Node<T>* add_temp_node(cplx location);

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
		void complete_triangles(Edge<T>* e);

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
		auto candidates_to_front_and_mark();

		// Reorders edges to put the candidates at the front, and marks the
		// edges of the triangles they belong to (visited_left and 
		// visted_right). returns iterator to the first non-candidate edge.
		auto mark_candidate_region_edges();

		// Does not preserve order of edges. If candidate edges have incomplete
		// triangles on either side, complete them recursively (i.e 
		// complete_triangles again for newly any generated candidates). Returns
		// an iterator to the end of the candidate edges.
		void extend_mesh();

		void clear_flags();
	private:
		// The order of edges and nodes is used in constructing the mesh,
		// but it will not be preserved afterward. Generally, traversing the
		// mesh will be done with the pointers in each mesh and node via member
		// functions.
		std::vector<std::unique_ptr<Node<T>>> nodes;
		std::vector<std::unique_ptr<Edge<T>>> edges;
		// When extending the mesh, new nodes and edges go here. After extension
		// is complete, these will be moved to the back of the main vectors.
		// Theoretically it's a bit faster to do it this way.
		std::vector<std::unique_ptr<Node<T>>> temp_nodes;

		std::function<cplx(cplx)> f;
		T precision;
	};

	template<typename T>
	inline Mesh<T>::Mesh(cplx corner1, cplx corner2, T edge_width, T final_prec,
		std::function<cplx(cplx)> func)
		: precision(final_prec), f(func)
	{
		// ensures corner1 is the top left and corner2 is bottom right.
		if (corner2.real() < corner1.real())
			std::swap(reinterpret_cast<T(&)[2]>(corner1)[0],
				reinterpret_cast<T(&)[2]>(corner2)[0]);
		if (corner2.imag() > corner1.imag())
			std::swap(reinterpret_cast<T(&)[2]>(corner1)[1],
				reinterpret_cast<T(&)[2]>(corner2)[1]);

		T width = corner2.real() - corner1.real();
		T height = corner1.imag() - corner2.imag();

		int col_count = ceil(width / edge_width);

		// The mesh will be constructed as an equilateral triangular grid, so we
		// need the height of the triangles.
		T row_width = edge_width * sin(M_PI / 3);

		int row_count = 1 + (int)ceil(height / row_width);
		int node_count = row_count * col_count;
		// 100 is an arbitrary bit of extra space for new points in
		// adapted mesh.
		nodes.reserve(node_count + 100);
		// Likewise, 200 is arbitrary.
		edges.reserve(3 * node_count - 2 * (row_count + col_count) + 1 + 200);

		T half_edge_width = edge_width / 2;
		T loc_y = corner1.imag();
		T loc_x;

		// create the nodes
		for (int y = 0; y < row_count; y++)
		{
			if (y % 2) loc_x = corner1.real() - half_edge_width;
			else loc_x = corner1.real();

			for (int x = 0; x < col_count; x++)
			{
				add_node(cplx(loc_x, loc_y));
				loc_x += edge_width;
			}

			loc_y -= row_width;
		}

		// Lambdas for readability in the following get_node connection section.

		auto connect_R = [=](int x, int y)
		{
			connect(nodes[y * col_count + x].get(),
				nodes[y * col_count + x + 1].get(),
				Direction::right);
		};

		auto connect_DL_even = [=](int x, int y)
		{
			connect(nodes[y * col_count + x].get(),
				nodes[(y + 1) * col_count + x].get(),
				Direction::down_left);
		};

		auto connect_DL_odd = [=](int x, int y)
		{
			connect(nodes[y * col_count + x].get(),
				nodes[(y + 1) * col_count + x - 1].get(),
				Direction::down_left);
		};

		auto connect_DR_even = [=](int x, int y)
		{
			connect(nodes[y * col_count + x].get(),
				nodes[(y + 1) * col_count + x + 1].get(),
				Direction::down_right);
		};

		auto connect_DR_odd = [=](int x, int y)
		{
			connect(nodes[y * col_count + x].get(),
				nodes[(y + 1) * col_count + x].get(),
				Direction::down_right);
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
		for (y = 0; y < row_count - 1; y++)
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

		extend_mesh();
	}

	template<typename T>
	inline void Mesh<T>::add_node(cplx location)
	{
		nodes.push_back(std::make_unique<Node<T>>(location));
		nodes.back()->calc_q(f(location));
	}

	template<typename T>
	inline Node<T>* Mesh<T>::add_temp_node(cplx location)
	{
		Node<T>* res;
		auto itr = std::find_if(temp_nodes.begin(), temp_nodes.end(),
			[&](auto&& N)
			{
				return abs(N->location - location) < precision / 100;
			});
		if (itr == temp_nodes.end())
		{
			temp_nodes.push_back(std::make_unique<Node<T>>(location));
			res = temp_nodes.back().get();
			res->calc_q(f(location));
		}
		else
			res = (*itr).get();

		return res;
	}

	template<typename T>
	inline Node<T>* Mesh<T>::split(Edge<T>* edge, T t)
	{
		add_node(t * edge->get_node(0)->location + (1 - t) *
			edge->get_node(1)->location);
		connect(edge->get_node(1), nodes.back().get(), edge->get_dir() + 3);
		edge->set_node(1, nodes.back().get());
		edge->split = true;
		edges.back()->split = true;
		edges.back()->visit_left = edge->visit_left;
		edges.back()->visit_right = edge->visit_right;
		return nodes.back().get();
	}

	template<typename T>
	inline void Mesh<T>::connect(Node<T>* n1, Node<T>* n2, int dir)
	{
		edges.push_back(std::make_unique<Edge<T>>(n1, n2, mod(dir, 6)));
		n1->set_edge(edges.back().get(), dir);
		n2->set_edge(edges.back().get(), dir + 3);
	}

	template<typename T>
	inline void Mesh<T>::complete_triangles(Edge<T>* e)
	{
		auto add_edges = [&](bool right) {
			int n0 = 0;
			int n1 = 1;
			int dir_change1 = 1;
			int dir_change2 = 2;
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
					complete_triangles(e);
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

				connect(e->get_node(n_0), N, e->get_dir() + neg * dir_change);
				recurse_if_candidate(edges.back().get());
			};

			// If neither edge of the triangle on the target side exists, then
			// add a node (or find one in temp_nodes) and create the edges.
			if (!e->get_next_CCW(n1) && !e->get_next_CW(n0))
			{
				// places the temp node by taking e's node 0's location and
				// adding a number with the same length as e, rotated
				// appropriately.
				auto N = add_temp_node(e->get_node(n0)->location + e->length()
					* exp((e->get_dir() + dir_change1 * neg)
						* M_PI / 3 * cplx(0, 1)));

				connect(e->get_node(n0), N, e->get_dir() + neg * dir_change1);
				edges.back()->length();
				recurse_if_candidate(edges.back().get());
				connect(e->get_node(n1), N, e->get_dir() + neg * dir_change2);
				edges.back()->length();
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
	inline std::vector<std::pair<std::complex<T>, int>>
		Mesh<T>::find_zeros_and_poles()
	{
		auto candidates_end = mark_candidate_region_edges();

		// Follows each edge around its loop, marking each edge traversed. If 
		// the path interects itself, follow each branch recursively. When it
		// cannot find a non-visited edge, the loop should be complete. The loop
		// estimates the location of a zero/pole by averaging the boundary
		// points, and the order by adding up the dq values for each edge
		// (4 quadrants = 1 full turn).

		std::vector<std::pair<cplx, int>> points;
		std::for_each(edges.begin(), candidates_end, [&](auto&& e)
			{
				cplx interior_point = (e->get_node(0)->location
					+ e->get_node(1)->location) / 2.0;
				T total_angle = 0;
				int boundarysize = 0;
				float sum_dq = 0;
				cplx node_sum = 0;

				// TODO: order calculations are inconsistent, maybe due to
				// differing ways to traverse self-intersecting loops. Must Fix.

				std::function<void(Edge<T>*)> next_boundary =
				[&](Edge<T>* edge)
				{
					auto N = edge->get_node(1);
					for (int i = 0; i < 6; i++)
					{
						if (auto next = N->get_edge(i); next
							&& next->visit_left ^ next->visit_right
							&& next->boundary == false)
						{
							// Reorients the edges so they form a directed loop.
							if (next->get_node(1) == N)
								next->reverse();
							boundarysize++;
							next->boundary = true;
							// angle must be added up to determine whether we
							// were traversing the loop clockwise or 
							// counterclockwise. positive = CCW, negative = CW.
							total_angle += angle(next->get_node(0)->location
								- interior_point, next->get_node(1)->location
								- interior_point);
							node_sum += next->get_node(0)->location
								+ next->get_node(1)->location;
							sum_dq += next->get_dq();
							next_boundary(next);
						}
					}
				};

				next_boundary(e.get());

				if (boundarysize)
				{
					if (total_angle < 0) sum_dq *= -1;
					points.push_back(std::make_pair(node_sum
						/ (cplx)(boundarysize * 2), sum_dq / 4));
					//std::cout << "boundary size: " << boundarysize << ", " <<
					//	"order: " << sum_dq / 4 << "\n";
				}
			});
		//for (auto&& P : points)
		//	std::cout << P.first << "\n";
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
				if (!E->split)
				{
					split(E);
				}
				return E->get_node(1);
			};

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
			if (!new_nodes[0].first->get_edge(dir))
				connect(new_nodes[0].first, new_nodes[1].first, dir);
			if (!new_nodes[3].first->get_edge(dir))
				connect(new_nodes[3].first, new_nodes[2].first, dir);
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
	inline auto Mesh<T>::candidates_to_front_and_mark()
	{
		return std::partition(edges.begin(), edges.end(),
			[](auto&& e)
			{
				if (abs(e->get_dq()) == 2)
				{
					e->boundary = true;
					return true;
				}
				else return false;
			});
	}

	template<typename T>
	inline auto Mesh<T>::mark_candidate_region_edges()
	{
		auto candidate_end = candidates_to_front_and_mark();

		// For each candidate edge, checks each edge belonging to a triangle
		// with the candidate (4 total). If one of those edges has been seen
		// both by a candidate to its left and a candidate to its right, it is
		// an interior edge and will be removed later. If it has been seen twice
		// from the same side and not from the other, it is still external.

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
	inline void Mesh<T>::extend_mesh()
	{
		auto C = candidates_to_front();
		auto dist = std::distance(edges.begin(), C);
		for (int i = 0; i < dist; i++)
		{
			auto e = edges[i].get();
			if (!(e->get_next_CW(0) && e->get_next_CCW(1) &&
				e->get_next_CCW(0) && e->get_next_CW(1))) complete_triangles(e);
		};
		nodes.insert(nodes.end(), std::make_move_iterator(temp_nodes.begin()),
			std::make_move_iterator(temp_nodes.end()));
		temp_nodes.clear();
	}

	template<typename T>
	inline void Mesh<T>::clear_flags()
	{
		std::for_each(/*std::execution::par_unseq,*/ edges.begin(), edges.end(),
			[](auto&& e)
			{
				e->boundary = false;
				e->split = false;
			});
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
		nodes[0]->set_edge(nullptr, direction);
		nodes[1]->set_edge(nullptr, direction + 3);
	}

	template<typename T>
	inline void Edge<T>::set_node(int i, Node<T>* N)
	{
		nodes[i] = N;
		N->set_edge(this, direction + i * 3);
		calc_dq();
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
		direction += 3;
		direction %= 6;
		dq *= -1;
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
			std::function<std::complex<T>(std::complex<T>)> f, T initial_mesh_len)
	{
		// Arbitrary. Later will add a trial-and error mesh sizer which tests
		// a few values in the desired neighborhood and picks the one which
		// finds the most candidate regions.
		if (initial_mesh_len < 0)
			initial_mesh_len = std::min(abs(ULcorner.real() - LRcorner.real())
				/ 20.1, abs(ULcorner.imag() - LRcorner.imag()) / 20.1);
		int iterations = std::log2(initial_mesh_len / precision);
		Mesh<double> mesh(ULcorner, LRcorner, initial_mesh_len, precision, f);

		for (int i = 0; i < iterations; i++)
		{
			mesh.adapt_mesh();
			mesh.clear_flags();
		}
		mesh.adapt_mesh();
		return mesh.find_zeros_and_poles();
	}
}
