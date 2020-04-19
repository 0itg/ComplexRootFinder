#include "zf.h"

typedef std::complex<double> cplx;
int main()
{
	auto points = zf::solve<cplx>(cplx(-1.8, 1.5), cplx(1.20, -1.20),
		0.00000000001,
		[](const cplx& z)
		{
			//return ((z - 1.0) * (z - cplx(0, 1)) * (z - cplx(0, 1)) * (z + 1.0)
			//	* (z + 1.0) * (z + 1.0) * (z + 1.0)) / ((z + cplx(0, 1))
			//		* (z + cplx(0.5, 0.5)) * (z + cplx(0.5, 0.5)));
			return (z * z + z + 1.0) / ((1.0+sin(z)) * sin(z));
		});
	for (auto&& P : points)
	{
		std::cout << P.first << ", " << "order: " << P.second << "\n";
	}
	return 0;
}