//
// Created by micah on 12/13/17.
//

#ifndef QUOTEFLOW_FULLYCONNECTEDNEURALNET_H
#define QUOTEFLOW_FULLYCONNECTEDNEURALNET_H

#include "../eigen/Eigen/Dense"
#include <random>

namespace NeuralNets {
	class Tanh {
	public:
		double Evaluate(double input) {
			return tanh(input);
		}

		double Derivative(double input) {
			return 1 - pow(tanh(input), 2);
		}
	};

	class ReLU {
	public:
		double Evaluate(double input) {
			return (input > 0) ? input : 0;
		}

		double Derivative(double input) {
			return (input > 0) ? 1 : 0;
		}

	};

	class SoftPlus {
	public:
		double Evaluate(double input) {
			return log(1 + exp(input));
		}

		double Derivative(double input) {
			double exp_i = exp(input);
			return exp_i / (1 + exp_i);
		}
	};

	template<class operation>
	class Evaluator {
	public:
		operation holder;
		const double operator()(const double value) {
			return holder.Evaluate(value);
		}
	};

	template<class operation>
	class Differentiator {
	public:
		operation holder;
		const double operator()(const double value) {
			return holder.Derivative(value);
		}
	};

	template<class num_values>
	class EuclideanLoss {
	public:
		Eigen::Matrix<double, num_values::value, 1> gradient;

		double Loss(Eigen::Matrix<double, num_values::value, 1> &true_value, Eigen::Matrix<double, num_values::value, 1> &predicted_value) {
			return (true_value - predicted_value).norm();
		}

		Eigen::Matrix<double, num_values::value, 1>& Gradient(Eigen::Matrix<double, num_values::value, 1> &true_value, Eigen::Matrix<double, num_values::value, 1> &predicted_value) {
			gradient = 2 * (true_value - predicted_value);
		};
	};


//	The true value specified for identity loss is interpreted as the total loss value
	template<class num_values>
	class IdentityLoss {
	public:
		Eigen::Matrix<double, num_values::value, 1> gradient;

		double Loss(Eigen::Matrix<double, num_values::value, 1> &true_value, Eigen::Matrix<double, num_values::value, 1> &predicted_value) {
			return true_value.sum();
		}

		Eigen::Matrix<double, num_values::value, 1>& Gradient(Eigen::Matrix<double, num_values::value, 1> &true_value, Eigen::Matrix<double, num_values::value, 1> &predicted_value) {
			gradient = true_value;
		};
	};


	template<class size, Eigen::Matrix<double, size::value, 1>& value_p>
	class MatrixWrapper{
	public:
		static const Eigen::Matrix<double, size::value, 1>& value { value_p };
	};


//	enum LayerType { InputLayer, LossLayer, FullyConnectedLayer};

//		General Form Layer

	template<class activation_function, class in_nodes, class out_nodes>
	class FullyConnectedLayer {
	public:
		typedef in_nodes input_nodes;
		typedef out_nodes output_nodes;

		Eigen::Matrix<double, output_nodes::value, input_nodes::value> weight_matrix;
		Eigen::Matrix<double, output_nodes::value, 1> output_vector;
		const Eigen::Matrix<double, input_nodes::value, 1>& input_vector;
		Eigen::Matrix<double, output_nodes::value, 1> ip_vector;
		Eigen::Matrix<double, input_nodes::value, 1> bp_errors;

		Evaluator<activation_function> evaluator;
		Differentiator<activation_function> differentiator;

		FullyConnectedLayer(const Eigen::Matrix<double, input_nodes::value, 1>& input) : input_vector(input){};

		Eigen::Matrix<double, output_nodes::value, 1>& Evaluate() const {
			ip_vector.noalias() = (weight_matrix * input_vector);
			output_vector.noalias() = ip_vector.unaryExpr(evaluator);
		};

		Eigen::Matrix<double, input_nodes::value, 1>& Backpropagate (double learning_rate, const Eigen::Matrix<double, output_nodes::value, 1> &input) const {
//				weight update
			ip_vector = ip_vector.unaryExpr(differentiator);
			bp_errors = weight_matrix.transpose()*((ip_vector.array()*input.array()).matrix());
			Eigen::Matrix<double, output_nodes::value, input_nodes::value> weight_gradient = learning_rate*(input.array()*ip_vector.array()).matrix()*input.transpose();
			weight_matrix -= weight_gradient;
			return bp_errors;
		}

	};

//		Input Layer
	template<class Nodes>
	class InputLayer {
	public:

		typedef Nodes input_nodes;
		typedef Nodes output_nodes;


		const Eigen::Matrix<double, Nodes::value, 1>& output_vector;

		InputLayer(const Eigen::Matrix<double, Nodes::value, 1>& input) : output_vector(input){};

		Eigen::Matrix<double, Nodes::value, 1>& Evaluate() const {
			return output_vector;
		};

		Eigen::Matrix<double, Nodes::value, 1>& Backpropagate(double learning_rate, const Eigen::Matrix<double, Nodes::value, 1> &input) const {
			return input;
		}
	};

//		Loss Layer

	template<class loss_function, class nodes_in>
	class LossLayer {
	public:
		typedef nodes_in input_nodes;
		typedef nodes_in output_nodes;

		const Eigen::Matrix<double, nodes_in::value, 1>& output_vector;
		Eigen::Matrix<double, nodes_in::value, 1> bp_errors;

		loss_function loss;

		LossLayer(const Eigen::Matrix<double, nodes_in::value, 1>& input) : output_vector(input){};

		Eigen::Matrix<double, nodes_in::value, 1>& Evaluate() const {};

		Eigen::Matrix<double, nodes_in::value, 1>& Backpropagate(double learning_rate, const Eigen::Matrix<double, nodes_in::value, 1>& ground_truth) const {
			return loss.Gradient(ground_truth, output_vector);
		};
	};



	template<class ... layerstype>
	class NeuralNet {

		template<class ... Args>
		class layer_stack_inside{};


		template<class layer_type, class ... rest>
		class layer_stack_inside<layer_type, rest...> {
		public:
			layer_type owned_layer;

			typedef layer_stack_inside<rest...> tail_stack_type;
			tail_stack_type tail_stack;

			typedef typename tail_stack_type::output_type output_type;

			layer_stack_inside(const Eigen::Matrix<double, layer_type::input_nodes::value, 1>& previous_layer_output) : owned_layer(previous_layer_output), tail_stack(owned_layer.output_vector) {};


			Eigen::Matrix<double, output_type::output_nodes::value, 1>& Evaluate() const {
				owned_layer.Evaluate();
				return tail_stack.Evaluate();
			};

			Eigen::Matrix<double, layer_type::input_nodes::value, 1> &Backpropagate(double learning_rate, Eigen::Matrix<double, layer_type::output_nodes::value, 1> &input) const {
				return owned_layer.Backpropagate(learning_rate, tail_stack.Backpropagate(learning_rate, input));
			};

		};

		template<class layer_type>
		class layer_stack_inside<layer_type> {
		public:

			typedef layer_type output_type;

			layer_type owned_layer;

			layer_stack_inside(const Eigen::Matrix<double, layer_type::input_nodes::value, 1>& previous_layer_output) : owned_layer(previous_layer_output) {};

			Eigen::Matrix<double, layer_type::output_nodes::value, 1>& Evaluate() const {
				owned_layer.Evaluate();
				return owned_layer.output_vector;
			};

			Eigen::Matrix<double, layer_type::input_nodes::value, 1> &Backpropagate(double learning_rate, Eigen::Matrix<double, layer_type::output_nodes::value, 1> &input) const {
				return owned_layer.Backpropagate(learning_rate, input);
			};
		};

		template<class input_layer, class ... layers>
		class LayerStack {
		public:
			typedef layer_stack_inside<input_layer, layers...> stacktype;
			stacktype stack;

			typedef typename stacktype::output_type::output_nodes output_size;
			typedef typename input_layer::input_nodes input_size;

			Eigen::Matrix<double, input_layer::input_nodes::value, 1> input_vector;

			LayerStack() : stack(input_vector) {};

			Eigen::Matrix<double, output_size::value, 1>& Evaluate(const Eigen::Matrix<double, input_layer::input_nodes::value, 1>& input) const {
				input_vector = input;
				return stack.Evaluate();
			}

			void Backpropagate(double learning_rate, const Eigen::Matrix<double, output_size::value, 1>& true_value) const {
				stack.Backpropagate(learning_rate, true_value);
			}

		};

//		Now for the Net itself!

		typedef LayerStack<layerstype...> stacktype;
		stacktype layers;

		Eigen::Matrix<double, stacktype::output_size::value, 1>& Evaluate(const Eigen::Matrix<double, stacktype::input_size::value , 1>& input) const {
			return layers.Evaluate(input);
		}

		void Learn(const Eigen::Matrix<double, stacktype::input_size::value, 1>& input_val, const Eigen::Matrix<double, stacktype::output_size::value, 1>& output_val, double learning_rate) const {
			layers.Evaluate(input_val);
			layers.Backpropagate(learning_rate, output_val);
		}

	};
}

#endif //QUOTEFLOW_FULLYCONNECTEDNEURALNET_H
