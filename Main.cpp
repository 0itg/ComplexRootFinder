#include "Node.h"

typedef std::complex<double> cplx;
int main()
{
	Mesh<double> test(cplx(-10.5, -10.2), cplx(10.2, 10.5), 0.3, 1,
		[](const cplx& z)
		{
		return (z - 1.0) * (z - cplx(0, 1)) * (z - cplx(0, 1)) * (z + 1.0) *
			(z + 1.0) * (z + 1.0) / (z + cplx(0, 1));
		});
	//test.apply_all();
	//test.apply_all([](const cplx& z) { return (z * z) + cplx(0.1,0.1); });
	test.adapt_mesh();
	test.find_zeros_and_poles();
	return 0;
}