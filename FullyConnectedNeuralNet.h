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


	template<class size, Eigen::Matrix<double, size::value, 1>& value>
	class MatrixWrapper{
	public:
		Eigen::Matrix<double, size::value, 1>& value;
	};


	template<class dimensions, class ... rem>
	class NeuralNet {

//		General Form Layer

		template<class activation_function, class nodes_in, class nodes_out, Eigen::Matrix<double, nodes_in::value, 1>& input_vector>
		class Layer {
		public:
			typedef nodes_out Nodes;
			Eigen::Matrix<double, nodes_out::value, nodes_in::value> weight_matrix;
			Eigen::Matrix<double, nodes_out::value, 1> output_vector;
			Eigen::Matrix<double, nodes_out::value, 1> ip_vector;
			Eigen::Matrix<double, nodes_in::value, 1> bp_errors;

			Evaluator<activation_function> evaluator;
			Differentiator<activation_function> differentiator;


			void Evaluate() {
				ip_vector.noalias() = (weight_matrix * input_vector);
				output_vector.noalias() = ip_vector.unaryExpr(evaluator);
			};

			Eigen::Matrix<double, nodes_in::value, 1>& Backpropagate(double learning_rate, Eigen::Matrix<double, nodes_out::value, 1> &input) {
//				weight update
				ip_vector = ip_vector.unaryExpr(differentiator);
				bp_errors = weight_matrix.transpose()*((ip_vector.array()*input.array()).matrix());
				Eigen::Matrix<double, nodes_out::value, nodes_in::value> weight_gradient = learning_rate*(input.array()*ip_vector.array()).matrix()*input_vector.transpose();
				weight_matrix -= weight_gradient;
				return bp_errors;
			}

		};

//		Input Layer
		template<class Nodes>
		class Layer {
		public:
			Eigen::Matrix<double, Nodes::value, 1> output_vector;

			void Evaluate() {};

			Eigen::Matrix<double, Nodes::value, 1>& Backpropagate(double learning_rate, Eigen::Matrix<double, Nodes::value, 1> &input) {
				return input;
			}
		};

		template <class Nodes>
		using InputLayer = Layer<Nodes>;

//		Loss Layer

		template<class loss_function, class nodes_in, Eigen::Matrix<double, nodes_in::value, 1>& input_vector>
		class Layer {
		public:
			typedef nodes_in Nodes;

			Eigen::Matrix<double, nodes_in::value, 1>& output_vector;
			Eigen::Matrix<double, nodes_in::value, 1> bp_errors;

			loss_function<nodes_in> loss;

			Layer() : output_vector(input_vector){};

			void Evaluate() {};

			Eigen::Matrix<double, nodes_in::value, 1>& Backpropagate(double learning_rate, Eigen::Matrix<double, nodes_in::value, 1>& ground_truth) {
				return loss.Gradient(ground_truth, input_vector);
			};


		};

		template <class Nodes, class loss_function, Eigen::Matrix<double, Nodes::value, 1>& input_vector>
		using LossLayer = Layer<loss_function, Nodes, input_vector>;



		template<Layer previous_layer, class loss, class nodes>
		class layer_stack_inside {
		public:
			typedef nodes Nodes;
			typedef typeof(previous_layer) previous_layer_type;
			typedef previous_layer_type::Nodes input_nodes;

			LossLayer<input_nodes, loss, previous_layer.output_vector> stack;
			typedef nodes forward_size;
			typedef MatrixWrapper<stack.output_vector> output;

			void Evaluate() {
				stack.Evaluate();
			};

			Eigen::Matrix<double, input_nodes::value, 1> &Backpropagate(double learning_rate, Eigen::Matrix<double, Nodes::value, 1> &input) {
				return stack.Backpropagate(learning_rate, input);
			};
		};


		template<Layer previous_layer, class activation, class nodes, class ... rest>
		class layer_stack_inside {
		public:
			typedef nodes Nodes;
			typedef typeof(previous_layer) previous_layer_type;
			typedef previous_layer_type::Nodes input_nodes;

			typedef Layer<activation, input_nodes, Nodes, previous_layer.output_vector> owned_layer_type;
			owned_layer_type owned_layer;

			typedef layer_stack_inside<owned_layer, rest...> tail_stack_type;
			tail_stack_type tail_stack;
			typedef tail_stack_type::forward_size forward_size;
			typedef tail_stack_type::output output;


			void Evaluate() {
				owned_layer.Evaluate();
				tail_stack.Evaluate();
			};

			Eigen::Matrix<double, input_nodes::value, 1> &Backpropagate(double learning_rate, Eigen::Matrix<double, Nodes::value, 1> &input) {
				return owned_layer.Backpropagate(learning_rate, tail_stack.Backpropagate(learning_rate, input));
			};

		};

		template<class input_nodes, class ... rest>
		class layer_stack_inside {
		public:
			typedef input_nodes Nodes;

			Layer<input_nodes> owned_layer;

			typedef layer_stack_inside<owned_layer, rest...> tail_stack_type;
			typedef tail_stack_type::forward_size forward_size;
			typedef tail_stack_type::output output;

			tail_stack_type tail_stack;

			void Evaluate() {
				owned_layer.Evaluate();
				tail_stack.Evaluate();
			};

			Eigen::Matrix<double, input_nodes::value, 1> &Backpropagate(double learning_rate, Eigen::Matrix<double, Nodes::value, 1> &input) {
				return tail_stack.Backpropagate(learning_rate, input);
			};


		};


		template<class input_nodes, class ... layers>
		class LayerStack {
		public:
			typedef layer_stack_inside<input_nodes, layers...> stacktype;
			stacktype stack;

			typedef stacktype::forward_size output_size;
			typedef stacktype::output output;

			output output_value;

			Eigen::Matrix<double, output_size::value, 1>& Evaluate(){
				stack.Evaluate();
				return output_value.value;
			}

			void Backpropagate(double learning_rate, Eigen::Matrix<double, output_size::value, 1>& true_value){
				stack.Backpropagate(learning_rate, true_value);
			}

			Eigen::Matrix<double, input_nodes::value, 1>& GetInput(){
				return stack.owned_layer.output_vector;
			}


		};

//		Now for the Net itself!

		typedef LayerStack<dimensions, rem...> layerstype;
		layerstype layers;

		Eigen::Matrix<double, dimensions::value, 1>& GetInput(){
			return layers.GetInput();
		};

		Eigen::Matrix<double, layerstype::output_size::value, 1>& GetOutput(){
			return layers.output_value.value;
		};

		Eigen::Matrix<double, layerstype::output_size::value, 1>& Evaluate(Eigen::Matrix<double, dimensions::value, 1>& input_val) {
			GetInput() = input_val;
			layers.Evaluate();
			return GetOutput();
		}

		void Learn(Eigen::Matrix<double, dimensions::value, 1>& input_val, Eigen::Matrix<double, layerstype::output_size::value, 1>& output_val, double learning_rate){
			GetInput() = input_val;
			layers.Evaluate();
			layers.Backpropagate(learning_rate, output_val);
		}

	};
}

#endif //QUOTEFLOW_FULLYCONNECTEDNEURALNET_H
