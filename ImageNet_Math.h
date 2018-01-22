#include "opencv.hpp"
#include <random>
#include <chrono>

#define __Mathlib

using namespace cv;
using namespace std;

namespace ImgNet_Math
{
	class Rand_uniform_Int
	{
	public:
		Rand_uniform_Int(int low, int high)
		{
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);
			std::uniform_int_distribution<int> distribution(low, high);
			r = bind(distribution, generator);
		}
		double operator()() { return r(); }
	private:
		function<int()> r;
	};

	class Rand_uniform_double
	{
	public:
		Rand_uniform_double(double low, double high)
		{
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);
			std::uniform_real_distribution<double> distribution(low, high);
			r = bind(distribution, generator);
		}
		double operator()() { return r(); }
	private:
		function<double()> r;
	};

	vector<double> uchar2dbl(vector<uchar>& input_vector);

	void makePair(vector<vector<uchar>>& v, vector<vector<uchar>>& w, vector<pair<vector<uchar>, vector<uchar>>> & result);

	vector<int> inRandomOrder(const vector<pair<vector<uchar>, vector<uchar>> >& data);

	void vector_length_queal(const vector<vector<uchar>>& v, const vector<vector<uchar>> & w);

	template<typename T, typename U>
	void vector_length_queal(const vector<T>& v, const vector<U> & w);

	void vector_length_security(const vector<double>& v, const vector<uchar>& w);

	void vector_subtract(const vector<double>& v, const vector<uchar>& w, vector<double>& result);

	template<typename T, typename U>
	double dot(const vector<T>& v, const vector<U>& w);

	void scalar_multiply(double c, vector<double>& v);

	double sum_of_squares(const vector<double>& v);

	double vector_sum(const vector<double>& vec);

	void randomVector(vector<double>& w, double lo = -0.3, double hi = 0.3);

	double square_error(vector<vector<uchar>>& X, vector<uchar>& Y, vector<double>& w, const function<double(double)>& actF);

	double partial_difference_quotient(function<double(vector<double>&)> f, vector<double> v, int i, double h = 0.001);

	template<typename T, typename U>
	void estimate_gradient(function<double(vector<T>&, vector<U>&, T&)> target_f, T& v, T& gradient, vector<T>& X, vector<U>& Y, double h);

	void estimate_gradient(function<double(vector<double>&)> f, vector<double> v, vector<double>& gradient, double h = 0.001);
}