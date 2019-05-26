//
// Created by micah on 12/13/17.
//

#ifndef QUOTEFLOW_FULLYCONNECTEDNEURALNET_H
#define QUOTEFLOW_FULLYCONNECTEDNEURALNET_H

#include "../eigen/Eigen/Dense"
#include <random>
#include <csignal>
#include "../Processes/Processes.h"

namespace NeuralNets {
	class Tanh {
	public:
		const double Evaluate(const double& input) const{
			return tanh(input);
		}

		const double Derivative(const double& input) const {
			return 1 - pow(tanh(input), 2);
		}
	};

	class ReLU {
	public:
		const double Evaluate(const double& input) const{
			return (input > 0) ? input : 0;
		}

		const double Derivative(const double& input) const{
			return (input > 0) ? 1 : 0;
		}

	};

	class LeakyReLU {
	public:
		const double Evaluate(const double& input) const{
			return (input > 0) ? input : 0.01*input;
		}

		const double Derivative(const double& input) const{
			return (input > 0) ? 1 : 0.01;
		}

	};

	class SoftPlus {
	public:
		const double Evaluate(const double& input) const {
//			try {
				return (input>=0) ? input : (exp(input)-1);
//			}
//			catch(...){
//				std::cerr<<input<<'\n';
//				throw(SIGFPE);
//			}
		}

		const double Derivative(const double& input) const {
			return (input>=0) ? 1 : exp(input);
		}
	};

	class Identity {
	public:
		const double Evaluate(const double& input) const {
			return input;
		}

		const double Derivative(const double& input) const {
			return 1;
		}
	};

	template<class operation>
	class Evaluator {
	public:
		operation holder;
		const double operator()(const double& value) const{
			return holder.Evaluate(value);
		}
	};

	template<class operation>
	class Differentiator {
	public:
		operation holder;
		const double operator()(const double& value) const{
			return holder.Derivative(value);
		}
	};

	template<class num_values>
	class EuclideanLoss {
	public:
		Eigen::Matrix<double, num_values::value, 1> gradient;

		double Loss(Eigen::Matrix<double, num_values::value, 1> *true_value, Eigen::Matrix<double, num_values::value, 1> *predicted_value) {
			return (true_value - predicted_value).norm();
		}

		Eigen::Matrix<double, num_values::value, 1>* Gradient(Eigen::Matrix<double, num_values::value, 1> *true_value, Eigen::Matrix<double, num_values::value, 1> *predicted_value) {
			gradient = 2 * ( *predicted_value - *true_value);
			return &gradient;
		};
	};


//	The true value specified for identity loss is interpreted as the total loss value
	template<class num_values>
	class IdentityLoss {
	public:
		Eigen::Matrix<double, num_values::value, 1> gradient;

		double Loss(Eigen::Matrix<double, num_values::value, 1> *true_value, Eigen::Matrix<double, num_values::value, 1> *predicted_value) {
			return true_value.sum();
		}

		Eigen::Matrix<double, num_values::value, 1>* Gradient(Eigen::Matrix<double, num_values::value, 1> *true_value, Eigen::Matrix<double, num_values::value, 1> *predicted_value) {
			gradient = true_value;
		};
	};


	template<class size, Eigen::Matrix<double, size::value, 1>* value_p>
	class MatrixWrapper{
	public:
		static const Eigen::Matrix<double, size::value, 1>* value { value_p };
	};


//	enum LayerType { InputLayer, LossLayer, FullyConnectedLayer};

//		General Form Layer

	template<class activation_function, class in_nodes, class out_nodes>
	class FullyConnectedLayer {
	public:
		typedef in_nodes input_nodes;
		typedef out_nodes output_nodes;

		typedef std::integral_constant<bool,true> has_weights;

		UniformGenerator gen;

		double decay_rate;
		double epsilon;

		Eigen::Matrix<double, output_nodes::value, input_nodes::value> weight_matrix;
        Eigen::Matrix<double, output_nodes::value, input_nodes::value> weight_gradient;
		Eigen::Matrix<double, output_nodes::value, input_nodes::value> squared_gradient;
		Eigen::Matrix<double, output_nodes::value, input_nodes::value> update_exp;
		Eigen::Matrix<double, output_nodes::value, 1> output_vector;
		Eigen::Matrix<double, input_nodes::value, 1>* input_vector;
		Eigen::Matrix<double, output_nodes::value, 1> ip_vector;
		Eigen::Matrix<double, input_nodes::value, 1> bp_errors;

		Evaluator<activation_function> evaluator;
		Differentiator<activation_function> differentiator;

		FullyConnectedLayer(Eigen::Matrix<double, input_nodes::value, 1>* input) : gen(), input_vector(input){
			weight_matrix = weight_matrix.unaryExpr([this](const double& x){return gen.GetSample(2)-1;});
			weight_gradient = Eigen::Matrix<double, output_nodes::value, input_nodes::value>::Zero();
			squared_gradient = Eigen::Matrix<double, output_nodes::value, input_nodes::value>::Zero();
			update_exp = Eigen::Matrix<double, output_nodes::value, input_nodes::value>::Zero();
			decay_rate = 0.95;
			epsilon = 1e-8;
		};

		Eigen::Matrix<double, output_nodes::value, 1>* Evaluate() {
//			std::cerr<<*input_vector<<'\n';
//			std::cerr<<weight_matrix<<'\n';

			ip_vector = (weight_matrix * (*input_vector));
			output_vector = ip_vector.unaryExpr(evaluator);
			return &output_vector;
		};

		Eigen::Matrix<double, output_nodes::value, 1>* Forward() {
			return Evaluate();
		};


		Eigen::Matrix<double, input_nodes::value, 1>* Backpropagate (double learning_rate, const Eigen::Matrix<double, output_nodes::value, 1> *input, bool apply = false) {
//				weight update
			ip_vector = ip_vector.unaryExpr(differentiator);
//			std::cerr<<"IP: \n"<<ip_vector<<'\n';

			bp_errors = weight_matrix.transpose()*((ip_vector.array()*input->array()).matrix());
//			std::cerr<<"BPE: \n"<<bp_errors<<'\n';
//			std::cerr<<"IN: \n"<<*input<<'\n';
//			std::cerr<<"WEIGHT: \n"<<weight_matrix<<'\n';

			Eigen::Matrix<double, output_nodes::value, input_nodes::value> gradient = learning_rate*(ip_vector.array()*input->array()).matrix()*input_vector->transpose();

			squared_gradient = decay_rate*squared_gradient + (1 - decay_rate)*(gradient.array().pow(2)).matrix();

			auto update = -(((update_exp.array()+epsilon).sqrt())/((squared_gradient.array() + epsilon).sqrt())*(gradient.array())).matrix();
			update_exp = decay_rate*update_exp + (1-decay_rate)*update.array().pow(2).matrix();
//			std::cerr<<"Weight Gradient: \n"<<update<<'\n';
//			std::cerr<<"Raw Gradient: \n"<<gradient<<'\n';
//			std::cerr<<"Factor matrix: \n"<<((update_exp.array()+epsilon).sqrt())/((squared_gradient.array() + epsilon).sqrt())<<'\n';
//			std::cerr<<"Factor matrix: \n"<<((squared_gradient.array() + epsilon).sqrt())<<'\n';

			weight_matrix += update;
//			if(apply) {
//				weight_gradient = weight_gradient.unaryExpr([](const double& x){return (x>1) ? 1 : ((x<-1) ? -1 : x);});
//                weight_matrix -= weight_gradient;
////				std::cerr<<"Gradient: "<<weight_gradient<<'\n';
//                weight_gradient = Eigen::Matrix<double, output_nodes::value, input_nodes::value>::Zero();
//            }
//			else{
//
//			}
			return &bp_errors;
		}

		void DumpWeights() {
			std::cerr<<"---------------\n"<<weight_matrix<<"\n---------------\n";
		};

	};

//		Input Layer
	template<class Nodes>
	class InputLayer {
	public:

		typedef Nodes input_nodes;
		typedef Nodes output_nodes;

		typedef std::integral_constant<bool,false> has_weights;


		Eigen::Matrix<double, Nodes::value, 1>& output_vector;

		InputLayer(Eigen::Matrix<double, Nodes::value, 1>* input) : output_vector(*input){};

		Eigen::Matrix<double, Nodes::value, 1>* Evaluate() {
			return &output_vector;
		};

		Eigen::Matrix<double, Nodes::value, 1>* Forward() {
			return &output_vector;
		};

		Eigen::Matrix<double, Nodes::value, 1>* Backpropagate(double learning_rate, Eigen::Matrix<double, Nodes::value, 1> *input, bool apply = false) {
			return input;
		}

		void DumpWeights() {
			std::cerr<<"No Weights\n";
		};
	};

//		Dropout layer
	template<class Nodes, class dropout_rate>
	class DropoutLayer {
	public:

		typedef Nodes input_nodes;
		typedef Nodes output_nodes;

		typedef std::integral_constant<bool,false> has_weights;

		UniformGenerator gen;

		double dropout_pass;

		Eigen::Matrix<double, output_nodes::value, 1> output_vector;
		Eigen::Matrix<double, output_nodes::value, 1> dropout_vector;
		Eigen::Matrix<double, input_nodes::value, 1>* input_vector;

		DropoutLayer(Eigen::Matrix<double, input_nodes::value, 1>* input) : input_vector(input){
			dropout_pass = double(dropout_rate::value)/100;
		};

		Eigen::Matrix<double, Nodes::value, 1>* Forward() {
			for(int i = 0; i < input_nodes::value; i++){
				dropout_vector(i) = (gen.GetSample(1) < dropout_pass) ? 0 : 1;
			}
			output_vector = (input_vector->array()*dropout_vector.array()).matrix();
			return &output_vector;
		};

		Eigen::Matrix<double, Nodes::value, 1>* Evaluate() {
			return &output_vector;
		};

		Eigen::Matrix<double, Nodes::value, 1>* Backpropagate(double learning_rate, Eigen::Matrix<double, Nodes::value, 1> *input, bool apply = false) {
			return input;
		}

		void DumpWeights() {
			std::cerr<<"No Weights\n";
		};
	};
//		Loss Layer

	template<class loss_function, class nodes_in>
	class LossLayer {
	public:
		typedef nodes_in input_nodes;
		typedef nodes_in output_nodes;

		typedef std::integral_constant<bool, false> has_weights;

		Eigen::Matrix<double, nodes_in::value, 1> &output_vector;
		Eigen::Matrix<double, nodes_in::value, 1> bp_errors;

		loss_function loss;

		LossLayer(Eigen::Matrix<double, nodes_in::value, 1> *input) : output_vector(*input) {};

		Eigen::Matrix<double, nodes_in::value, 1>* Evaluate() {
			return &output_vector;
		};

		Eigen::Matrix<double, nodes_in::value, 1>* Forward() {
			return &output_vector;
		};

		Eigen::Matrix<double, nodes_in::value, 1> *
		Backpropagate(double learning_rate, Eigen::Matrix<double, nodes_in::value, 1> *ground_truth, bool apply = false) {
			loss.Gradient(ground_truth, &output_vector);
//			std::cerr<<"LOSS: "<<loss.gradient<<'\n';
//			std::cerr<<"TRUTH: "<<*ground_truth<<'\n';
//			std::cerr<<"Output: "<<output_vector<<'\n';

			return &loss.gradient;
		};

		void DumpWeights() {
			std::cerr << "No Weights\n";
		};

	};

	template<class ... layerstype>
	class NeuralNet {
	public:

		template<class ... Args>
		class layer_stack_inside{};


		template<class layer_type, class ... rest>
		class layer_stack_inside<layer_type, rest...> {
		public:
			layer_type owned_layer;

			typedef layer_stack_inside<rest...> tail_stack_type;
			tail_stack_type tail_stack;

			typedef typename tail_stack_type::output_type output_type;

			layer_stack_inside(Eigen::Matrix<double, layer_type::input_nodes::value, 1>* previous_layer_output) : owned_layer(previous_layer_output), tail_stack(&owned_layer.output_vector) {};


			Eigen::Matrix<double, output_type::output_nodes::value, 1>* Evaluate() {
				owned_layer.Evaluate();
				return tail_stack.Evaluate();
			};

			Eigen::Matrix<double, output_type::output_nodes::value, 1>* Forward() {
				owned_layer.Forward();
				return tail_stack.Forward();
			};

			Eigen::Matrix<double, layer_type::input_nodes::value, 1>* Backpropagate(double learning_rate, Eigen::Matrix<double, output_type::output_nodes::value, 1> *input, bool apply = false) {
				return owned_layer.Backpropagate(learning_rate, tail_stack.Backpropagate(learning_rate, input, apply), apply);
			};

			void DumpWeights() {
				owned_layer.DumpWeights();

				tail_stack.DumpWeights();
			}

		};

		template<class layer_type>
		class layer_stack_inside<layer_type> {
		public:

			typedef layer_type output_type;

			layer_type owned_layer;

			layer_stack_inside(Eigen::Matrix<double, layer_type::input_nodes::value, 1>* previous_layer_output) : owned_layer(previous_layer_output) {};

			Eigen::Matrix<double, layer_type::output_nodes::value, 1>* Evaluate() {
				owned_layer.Evaluate();
				return &owned_layer.output_vector;
			};

			Eigen::Matrix<double, layer_type::output_nodes::value, 1>* Forward() {
				owned_layer.Forward();
				return &owned_layer.output_vector;
			};

			Eigen::Matrix<double, layer_type::input_nodes::value, 1>* Backpropagate(double learning_rate, Eigen::Matrix<double, layer_type::output_nodes::value, 1>* input, bool apply = false) {
				return owned_layer.Backpropagate(learning_rate, input, apply);
			};

			void DumpWeights(){
				owned_layer.DumpWeights();
			}
		};

		template<class input_layer, class ... layers>
		class LayerStack {
		public:
			typedef layer_stack_inside<input_layer, layers...> stacktype;
			stacktype stack;

			typedef typename stacktype::output_type::output_nodes output_size;
			typedef typename input_layer::input_nodes input_size;

			mutable Eigen::Matrix<double, input_layer::input_nodes::value, 1> input_vector;

			LayerStack() : stack(&input_vector) {};

			Eigen::Matrix<double, output_size::value, 1>* Evaluate(Eigen::Matrix<double, input_layer::input_nodes::value, 1>* input) {
				input_vector = *input;
				return stack.Evaluate();
			}

			Eigen::Matrix<double, output_size::value, 1>* Forward(Eigen::Matrix<double, input_layer::input_nodes::value, 1>* input) {
				input_vector = *input;
				return stack.Forward();
			}

			void Backpropagate(double learning_rate, Eigen::Matrix<double, output_size::value, 1>* true_value, bool apply = false){
				stack.Backpropagate(learning_rate, true_value, apply);
			}

			void DumpWeights(){
				stack.DumpWeights();
			}
		};

//		Now for the Net itself!

		typedef LayerStack<layerstype...> stacktype;
		stacktype layers;

		Eigen::Matrix<double, stacktype::output_size::value, 1>* Evaluate(Eigen::Matrix<double, stacktype::input_size::value , 1>* input) {
			return layers.Evaluate(input);
		}

		Eigen::Matrix<double, stacktype::output_size::value, 1>* Forward(Eigen::Matrix<double, stacktype::input_size::value , 1>* input) {
			return layers.Forward(input);
		}

		void Learn(Eigen::Matrix<double, stacktype::input_size::value, 1>* input_val, Eigen::Matrix<double, stacktype::output_size::value, 1>* output_val, double learning_rate, bool apply = false) {

//			    feenableexcept(FE_ALL_EXCEPT);

			layers.Evaluate(input_val);
			layers.Backpropagate(learning_rate, output_val, apply);
//			    fedisableexcept(FE_ALL_EXCEPT);

		}

		void DumpWeights(){
			layers.DumpWeights();
		}
	};
}

#endif //QUOTEFLOW_FULLYCONNECTEDNEURALNET_H
