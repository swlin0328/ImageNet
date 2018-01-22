#include <Windows.h>
#include "ImageNet_Math.h"

#define __MLlib

using namespace cv;
using namespace std;
using namespace ImgNet_Math;
#pragma once
namespace ImgNet_ML
{
	//監督式學習-分類器介面
	template<typename T>
	class classifier
	{
	private:
		virtual double err_inPrediction(const vector<T>& Y) = 0;
		virtual double cost_inPrediction(const vector<T>& Y) = 0;

	public:
		virtual void train(vector<vector<T>>& X, vector<T>& y) = 0;
		virtual void classify(const vector<vector<T>>& X) = 0;
		virtual void show_validate_result(const vector<T>& Y) = 0;
		virtual double predict_prob(vector<T>& X) = 0;
		virtual void show_train_result() = 0;
		virtual ~classifier() {};
	};

	//感知器
	template<typename T>
	class perceptron : public virtual classifier<T>
	{
	private:
		double leraning_Rate;
		int n_iter;
		vector<double> w;
		vector<double> cost;
		vector<T> predict_y;
		virtual double err_inPrediction(const vector<T>& Y) override;
		virtual double cost_inPrediction(const vector<T>& Y) override;

	protected:
		void train_for_network(perceptron&, int, double);
		vector<double>& get_neuron_w(perceptron&);
		function<double(double)> actFn;

	public:
		virtual void train(vector<vector<T>>& X, vector<T>& y) override;
		virtual void classify(const vector<vector<T>>& X) override;
		virtual void show_validate_result(const vector<T>& Y) override;
		virtual void show_train_result() override;
		virtual double predict_prob(vector<T>& X) override;
		void train_clear(vector<double>& cost, vector<double>& w, int vSize);
		perceptron(double eta = 0.0001, int epoch = 500, const function<double(double)>& actFunction = activation_for_hyperbolic) :leraning_Rate(eta), n_iter(epoch), actFn(actFunction) {}
	};

	//神經網路
	template<typename T>
	class neuron_network : protected perceptron<T>
	{
	private:
		int n_iter;
		function<double(double)> actFn;
		double learning_rate;
		vector<vector<vector<perceptron<T>>>> NN;
		vector<perceptron<T>> make_input_layer(int input_dim, int row, double eta, int epoch, const function<double(double)>& actFunction);
		vector<perceptron<T>> make_hidden_layer(int row, double eta, int epoch, const function<double(double)>& actFunction);
		vector<perceptron<T>> make_output_layer(int output_dim, int row, double eta, int epoch, const function<double(double)>& actFunction);
		void make_Neural_Network(vector<vector<vector<perceptron<T>>>>& NN, int input_dim, int output_dim, int row, int col, int depth, double eta, int epoch, const function<double(double)>& actFunction);
		vector<vector<double>> feed_forward_2d(vector<T>&);
		void backpropagate_2d(vector<T>&, vector<T>&, int step, int pretrain = 50, double diff_h = 0.001);

	public:
		void train(vector<vector<T>>&, vector<vector<T>>&, int pretrain = 200, double precision = 0.01, double diff_h = 0.001);
		void predict(vector<vector<T>>& X, vector<vector<T>>& Y);
		neuron_network() = delete;
		neuron_network(int input_dim, int output_dim, int row = 5, int col = 5, int depth = 1, const function<double(double)>& actFunction = activation_for_logistic, int epoch = 10000, double NN_learning_rate = 0.1, double eta = 0.0001) : n_iter(epoch), learning_rate(NN_learning_rate), actFn(actFunction) { make_Neural_Network(NN, input_dim, output_dim, row, col, depth, eta, epoch, actFunction); }
		~neuron_network() { NN.~vector(); }
	};

	//特徵檢測
	Mat OLBP(Mat& srcImage);

	Mat regionExtraction(Mat& srcImage, int xRoi, int yRoi, int widthRoi, int heightRoi, bool open);

	//影像讀取
	void readImgNamefromFile(string folderName, vector<string>& imgPaths);

	//訓練函數
	double activation_for_hyperbolic(double wX);

	double square_error(vector<vector<uchar>>& X, vector<uchar>& Y, vector<double>& w, const function<double(double)>& actF);
	
	template<typename T>
	void perceptron<T>::train(vector<vector<T>>& X, vector<T>& y)
	{
		train_clear(cost, w, X[0].size());
		ImgNet_Math::randomVector(w);

		for (int i = 0; i < n_iter; i++)
		{
			vector<double> gradient;

			ImgNet_Math::estimate_gradient(
				[&](vector<double> w_0)
			{double output{ 0 }, err{ 0 };
			return square_error(X, y, w_0, actFn); }, w, gradient);

			for (int k = 0; k < w.size(); k++)
			{
				w[k] -= leraning_Rate * gradient[k];
			}
			cost.push_back(square_error(X, y, w, actFn));
			classify(X);
		}
	}

	template<typename T>
	void perceptron<T>::train_for_network(perceptron<T>& neuron, int index, double delta)
	{
		neuron.w[index] += delta;
	}

	template<typename T>
	vector<double>& perceptron<T>::get_neuron_w(perceptron<T>& neuron)
	{
		return neuron.w;
	}

	template<typename T>
	void perceptron<T>::classify(const vector<vector<T>>& X)
	{
		predict_y.clear();
		predict_y.resize(0);
		for (int i = 0; i < X.size(); i++)
		{
			double value = actFn(ImgNet_Math::dot(X[i], w));
			if (actFn(ImgNet_Math::dot(X[i], w)) > 0.5)
			{
				predict_y.push_back(1);
			}
			else
			{
				predict_y.push_back(-1);
			}
		}
	}

	template<typename T>
	void perceptron<T>::show_train_result()
	{
		cout << "Number of epoch: " << n_iter << " , the cost values for the each epoch in training\n";
		for (int i = 0; i < n_iter; i++)
		{
			if ((i + 1) % 50 == 0)
			{
				cout << "The epoch " << i + 1 << " of the cost is: " << cost[i] << "\n";
			}
		}
	}

	template<typename T>
	double perceptron<T>::predict_prob(vector<T>& input)
	{
		return ImgNet_Math::dot(input, w);
	}

	template<typename T>
	void perceptron<T>::show_validate_result(const vector<T>& Y)
	{
		double num = err_inPrediction(Y);
		double accurate = 1 - (num / Y.size());
		cout << "The predict accurate: " << setprecision(4) << 100 * accurate << " %\n";
	}

	template<typename T>
	double perceptron<T>::err_inPrediction(const vector<T>& Y)
	{
		int num = 0;
		for (int i = 0; i < Y.size(); i++)
		{
			int predict = predict_y[i] / abs(predict_y[i]);
			if (predict != Y[i])
			{
				num += 1;
			}
		}
		return num;
	}

	template<typename T>
	double perceptron<T>::cost_inPrediction(const vector<T>& Y)
	{
		double cost{ 0 };
		for (int i = 0; i < Y.size(); i++)
		{
			cost += pow((Y[i] - predict_y[i]), 2);
		}
		return cost;
	}

	template<typename T>
	void perceptron<T>::train_clear(vector<double>& cost, vector<double>& w, int vSize)
	{
		cost.clear();
		w.clear();
		cost.resize(0);
		w.resize(vSize, 0);
	}


	template<typename T>
	void neuron_network<T>::make_Neural_Network(vector<vector<vector<perceptron<T>>>>& NN, int input_dim, int output_dim, int row, int col, int depth, double eta, int epoch, const function<double(double)>& actFunction)
	{
		vector<vector<vector<perceptron<T>>>> network_3d;

		for (int k = 0; k < depth; k++)
		{
			vector<vector<perceptron<T>>> network_2d;
			network_2d.push_back(make_input_layer(input_dim, row, eta, epoch, actFunction));

			//hidden_layer
			for (int j = 1; j < col - 1; j++)
			{
				network_2d.push_back(make_hidden_layer(row, eta, epoch, actFunction));
			}
			network_2d.push_back(make_output_layer(output_dim, row, eta, epoch, actFunction));

			network_3d.push_back(network_2d);
		}
		NN = move(network_3d);
	}

	template<typename T>
	vector<perceptron<T>> neuron_network<T>::make_input_layer(int input_dim, int row, double eta, int epoch, const function<double(double)>& actFunction)
	{
		vector<perceptron<T>> input_layer;
		for (int i = 0; i < row; i++)
		{
			perceptron<T> neuron(eta, epoch, actFunction);
			vector<double>& w = get_neuron_w(neuron);
			w.resize(input_dim, 0);
			randomVector(w, -0.5, 0.5);
			input_layer.push_back(neuron);
		}
		return input_layer;
	}

	template<typename T>
	vector<perceptron<T>> neuron_network<T>::make_hidden_layer(int row, double eta, int epoch, const function<double(double)>& actFunction)
	{
		vector<perceptron<T>> hidden_layer;
		for (int i = 0; i < row; i++)
		{
			perceptron<T> neuron(eta, epoch, actFunction);
			vector<double>& w = get_neuron_w(neuron);
			w.resize(row + 1, 0);
			randomVector(w, -0.5, 0.5);
			hidden_layer.push_back(neuron);
		}
		return  hidden_layer;
	}

	template<typename T>
	vector<perceptron<T>> neuron_network<T>::make_output_layer(int output_dim, int row, double eta, int epoch, const function<double(double)>& actFunction)
	{
		vector<perceptron<T>> output_layer;
		for (int i = 0; i < output_dim; i++)
		{
			perceptron<T> neuron(eta, epoch, actFunction);
			vector<double>& w = get_neuron_w(neuron);
			w.resize(row + 1, 0);
			randomVector(w, -0.5, 0.5);
			output_layer.push_back(neuron);
		}
		return output_layer;
	}

	template<typename T>
	vector<vector<double>> neuron_network<T>::feed_forward_2d(vector<T>& input_vector)
	{
		vector<vector<double>> output_each_layers;
		vector<double> inputData = uchar2dbl(input_vector);
		output_each_layers.push_back(inputData);

		for (int i = 0; i < NN.size(); i++)
		{
			for (int j = 0; j < NN[i].size(); j++)
			{
				vector<double> score_output, actFn_output;
				if (j < NN[i].size() - 1)
				{
					score_output.push_back(1);
					actFn_output.push_back(1);
				}

				for (int neuron = 0; neuron < NN[i][j].size(); neuron++)
				{
					double score = dot(get_neuron_w(NN[i][j][neuron]), inputData);
					score_output.push_back(score);
					actFn_output.push_back(actFn(score));
				}
				output_each_layers.push_back(score_output);
				inputData = actFn_output;
			}
		}
		return output_each_layers;
	}

	template<typename T>
	void neuron_network<T>::backpropagate_2d(vector<T>& input_vector, vector<T>& target, int step, int pretrain, double diff_h)
	{
		vector<vector<double>> outputs{ feed_forward_2d(input_vector) };
		vector<vector<double>> layers_deltas;
		vector<double> output_deltas;
		int num_output_layer = outputs.size() - 1;
		int num_NN_layer = outputs.size() - 2;
		vector_length_queal(outputs[num_output_layer], target);
		//outlayer
		for (int i = 0; i < target.size(); i++)
		{
			double diff_actFn = (actFn(outputs[num_output_layer][i] + diff_h) - actFn(outputs[num_output_layer][i])) / diff_h;
			output_deltas.push_back(diff_actFn * (actFn(outputs[num_output_layer][i]) - target[i]));
		}
		layers_deltas.push_back(output_deltas);

		for (int i = 0; i < NN[0][num_NN_layer].size(); i++)
		{
			for (int j = 0; j < get_neuron_w(NN[0][num_NN_layer][i]).size(); j++)
			{
				double partial_gradient = learning_rate * output_deltas[i] * actFn(outputs[num_output_layer - 1][j]);
				if (step < pretrain || step > pretrain*(num_NN_layer + 1))
				{
					train_for_network(NN[0][num_NN_layer][i], j, -partial_gradient);
				}
			}
		}

		//hidden layer
		for (int num_layer = num_output_layer; num_layer > 1; num_layer--)
		{
			int NN_layer = num_layer - 1;
			vector<double> hidden_deltas;

			for (int i = 1; i < outputs[num_layer - 1].size(); i++)
			{
				double hidden_delta = 0;
				double diff_actFn = (actFn(outputs[num_layer - 1][i] + diff_h) - actFn(outputs[num_layer - 1][i])) / diff_h;
				for (int k = 0; k < layers_deltas[num_output_layer - num_layer].size(); k++)
				{
					hidden_delta += (diff_actFn * layers_deltas[num_output_layer - num_layer][k] * get_neuron_w(NN[0][NN_layer][k]).at(i));
				}
				hidden_deltas.push_back(hidden_delta);
			}
			layers_deltas.push_back(hidden_deltas);

			for (int i = 0; i < NN[0][NN_layer - 1].size(); i++)
			{
				for (int j = 0; j < get_neuron_w(NN[0][NN_layer - 1][i]).size(); j++)
				{
					double partial_gradient;
					if (num_layer - 2 > 0)
					{
						partial_gradient = learning_rate * hidden_deltas[i] * actFn(outputs[num_layer - 2][j]);
					}
					else
					{
						partial_gradient = learning_rate * hidden_deltas[i] * outputs[num_layer - 2][j];
					}
					if (step > pretrain*(num_NN_layer + 1) || ((num_NN_layer - NN_layer + 1)*pretrain > step && (num_NN_layer - NN_layer)*pretrain < step))
					{
						train_for_network(NN[0][NN_layer - 1][i], j, -partial_gradient);
					}
				}
			}
		}
	}

	template<typename T>
	void neuron_network<T>::train(vector<vector<T>>& input, vector<vector<T>>& target, int pretrain, double precision, double diff_h)
	{
		vector<pair<vector<T>, vector<T>>> dataset;
		makePair(input, target, dataset);
		auto rand_index_set = inRandomOrder(dataset);
		int count = 1, check_count = 1;

		for (int step = 0; step < n_iter; step++)
		{
			for (int index = 0; index < dataset.size(); index++)
			{
				backpropagate_2d(dataset[index].first, dataset[index].second, step, pretrain);
			}

			count++;
			check_count++;
			if (count % 200 == 0)
			{
				learning_rate *= 0.96;
				if (check_count % 1000 == 0)
				{
					Rand_uniform_Int rand_gen(0, rand_index_set.size() - 1);
					int index = rand_index_set[rand_gen()];

					vector<vector<double>> check_result = feed_forward_2d(dataset[index].first);
					vector<double> result = check_result.back();
					for (int i = 0; i < result.size(); i++)
					{
						result[i] = actFn(result[i]);
					}

					vector_subtract(result, dataset[index].second, result);
					if (sum_of_squares(result) < precision) { break; }
				}
			}
		}
	}

	template<typename T>
	void neuron_network<T>::predict(vector<vector<T>>& validata_X, vector<vector<T>>& validate_Y)
	{
		for (int i = 0; i < validata_X.size(); i++)
		{
			vector<vector<double>> output_each_layers = feed_forward_2d(validata_X[i]);
			vector<double> P_predict;
			int last_layer = output_each_layers.size() - 1;

			cout << "The predict probability for " << i << "th data is\n";
			for (int j = 0; j < output_each_layers[last_layer].size(); j++)
			{
				P_predict.push_back(actFn(output_each_layers[last_layer][j]));
			}
			for (int j = 0; j < P_predict.size(); j++)
			{
				ImgNet_Math::scalar_multiply(1 / ImgNet_Math::vector_sum(P_predict), P_predict);
				if (P_predict[j] < 0.01) { P_predict[j] = 0; }
				cout << P_predict[j] << "  ";
			}
			cout << "\n";
			cout << "The true answer for " << i << "th data is\n";
			for (int j = 0; j < validate_Y[i].size(); j++)
			{
				cout << validate_Y[i][j] << "  ";
			}
			cout << "\n\n\n";
		}
	}
}
