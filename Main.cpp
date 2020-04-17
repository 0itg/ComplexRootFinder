#include "Node.h"

typedef std::complex<double> cplx;
int main()
{
	auto points = zf::solve<double>(cplx(-1.80, 1.50), cplx(1.20, -1.20),
		0.0000001,
		[](const cplx& z)
		{
			return (z - 1.0) * (z - cplx(0, 1)) * (z - cplx(0, 1)) * (z + 1.0) *
				(z + 1.0) * (z + 1.0) / (z + cplx(0, 1));
			//return (z * z + z + 1.0) / (1.0 + std::sin(z));
		}, true, 0.29);
	for (auto&& P : points)
	{
		std::cout << P.first << ", " << "order: " << P.second << "\n";
	}
	return 0;
}