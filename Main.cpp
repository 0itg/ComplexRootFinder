#include "zf.h"
#include <iomanip>
#include <boost/multiprecision/number.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_complex.hpp>

typedef double data_t;
//typedef boost::multiprecision::
//	number<boost::multiprecision::cpp_bin_float<100>> data_t;
typedef std::complex<data_t> cplx;
//typedef boost::multiprecision::cpp_complex_quad cplx;
int main()
{
	auto points = zf::solve<cplx>(cplx(-1.8, 1.5), cplx(1.20, -1.20),
		data_t(0.00000001),
		[](const cplx& z)
		{
			return ((z - data_t(1.0)) * (z - cplx(0, 1))
				* (z - cplx(0, 1)) * (z + data_t(1.0))
				* (z + data_t(1.0)) * (z + data_t(1.0))
				* (z + data_t(1.0))) / (exp(z + cplx(0, 1))
				* (z + cplx(0.5, 0.5)) * (z + cplx(0.5, 0.5)));
			//return data_t(1.0) + sin(z);
		});
	for (auto&& P : points)
	{
		std::cout << std::setprecision(25)
			<< P.first << ", " << "order: " << P.second << "\n";
	}
	return 0;
}