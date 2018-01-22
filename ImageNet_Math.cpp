#include "ImageNet_Math.h"

namespace ImgNet_Math
{
	vector<double> uchar2dbl(vector<uchar>& input_vector)
	{
		vector<double> result;
		for (int i = 0; i < input_vector.size(); i++)
		{
			result.push_back(static_cast<double>(input_vector[i]));
		}
		return result;
	}

	void makePair(vector<vector<uchar>>& v, vector<vector<uchar>>& w, vector<pair<vector<uchar>, vector<uchar>>> & result)
	{
		vector_length_queal(v, w);
		for (int i = 0; i < v.size(); i++)
		{
			result.push_back(pair<vector<uchar>, vector<uchar>>{v[i], w[i]});
		}
	}

	vector<int> inRandomOrder(const vector<pair<vector<uchar>, vector<uchar>> >& data)
	{
		vector<int> indexes;
		for (int i = 0; i < data.size(); i++)
		{
			indexes.push_back(i);
		}
		unsigned seed = (unsigned)time(NULL);
		shuffle(indexes.begin(), indexes.end(), std::default_random_engine(seed));
		return indexes;
	}

	void vector_length_queal(const vector<vector<uchar>>& v, const vector<vector<uchar>> & w)
	{
		assert(v.size() == w.size());
	}

	template<typename T, typename U>
	void vector_length_queal(const vector<T>& v, const vector<U> & w)
	{
		assert(v.size() == w.size());
	}

	void vector_length_security(const vector<double>& v, const vector<uchar>& w)
	{
		assert(v.capacity() >= w.size());
	}

	void vector_subtract(const vector<double>& v, const vector<uchar>& w, vector<double>& result)
	{
		vector_length_queal(v, w);
		vector_length_security(result, w);

		for (int i = 0; i < v.size(); i++)
		{
			result[i] = v[i] - w[i];
		}
	}

	template<typename T, typename U>
	double dot(const vector<T>& v, const vector<U>& w)
	{
		vector_length_queal(v, w);
		double sum = 0;

		for (int i = 0; i < v.size(); i++)
		{
			sum += v[i] * w[i];
		}

		return sum;
	}

	void scalar_multiply(double c, vector<double>& v)
	{
		for (int i = 0; i < v.size(); i++)
		{
			v[i] *= c;
		}
	}

	double sum_of_squares(const vector<double>& v)
	{
		return dot(v, v);
	}

	double vector_sum(const vector<double>& vec)
	{
		double sum = 0;
		for (int i = 0; i < vec.size(); i++)
		{
			sum += vec[i];
		}
		return sum;
	}

	void randomVector(vector<double>& w, double lo, double hi)
	{
		Rand_uniform_double Random(lo, hi);
		for (int i = 0; i < w.size(); i++)
		{
			w[i] = Random();
		}
	}

	double square_error(vector<vector<uchar>>& X, vector<uchar>& Y, vector<double>& w, const function<double(double)>& actF)
	{
		double errValue = 0;
		for (int i = 0; i < X.size(); i++)
		{
			errValue += pow((Y[i] - actF(dot(X[i], w))), 2);
		}
		return errValue;
	}

	double partial_difference_quotient(function<double(vector<double>&)> f, vector<double> w, int i, double h)
	{
		vector<double> v = w;
		v[i] += h;
		return (f(v) - f(w)) / h;
	}

	template<typename T, typename U>
	void estimate_gradient(function<double(vector<T>&, vector<U>&, T&)> target_f, T& v, T& gradient, vector<T>& X, vector<U>& Y, double h)
	{
		vector<double> result(v.size(), 0);

		for (int i = 0; i < v.size(); i++)
		{
			result[i] = partial_difference_quotient(target_f, v, X, Y, i, h);
		}

		gradient = result;
	}

	void estimate_gradient(function<double(vector<double>&)> f, vector<double> v, vector<double>& gradient, double h)
	{
		vector<double> result(v.size(), 0);

		for (int i = 0; i < v.size(); i++)
		{
			result[i] = partial_difference_quotient(f, v, i, h);
		}
		gradient = result;
	}
}