/*Deep Euler implementation of a sonochemical bubble model*/

#include <iostream>
#include <fstream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

#include <torch/script.h>
#include <Eigen/Core>
#include <boost/numeric/odeint.hpp>

const double rho_L = 9.970639504998557e+02;
const double p_inf = 1.0e+5;
const double sigma = 0.071977583160056;
const double gamma = 1.33;
const double c_L = 1.497251785455527e+03; // water 25 Celsius
const double mu_L = 8.902125058209557e-04; //25 Celsius
const double lambda = 0.6084; //water 25 Celsius
const double T_inf = 298.15; // 25 Celsius

const double R_E = 10e-6; //1...10u
const double p_A = 0.5e5; //0.5..2 bar
const double f = 100e3; //20 kHz... 2 MHz

using namespace std;

const int N = 16;
const int N_z = N / 2 - 1;

const int nn_inputs = 2 + 3 + N_z;
const int nn_outputs = 3 + N_z;
const int system_order = nn_outputs + 1;
c10::TensorOptions global_tensor_op;

//Modify these to load the correct model
string file_name = "../simulations/bub_dem_1e-5.txt";
string model_file = "../../../training/traced_model_bub1_e185_2111101855.pt";
string scaler_file = "../../../training/scaler_bub1_2111101855.psca";

typedef double value_type;
typedef vector<value_type> state_type;
typedef Eigen::Matrix<value_type, N / 2, N / 2> matrix_type;

struct std_scaler {
	torch::Tensor mean;
	torch::Tensor scale;

	torch::Tensor operator()(torch::Tensor tensor) {
		return (tensor - mean) / scale;
	}
	torch::Tensor inverse_transform(torch::Tensor tensor) {
		return tensor * scale + mean;
	}
	void parse(istream& is, int numel) {
		mean = torch::ones({ 1, numel });
		is.get();
		double temp = 0.0;
		for (int i = 0; i < numel; i++) {
			is >> temp;
			mean[0][i] = temp;
		}
		is.get();
		is.get();
		scale = torch::ones({ 1, numel });
		is.get();
		for (int i = 0; i < numel; i++) {
			is >> temp;
			scale[0][i] = temp;
		}
		is.get();
		is.get();
	}
};

struct norm_scaler {
	torch::Tensor data_min;
	torch::Tensor data_max;
	double min = 0;
	double max = 0;

	torch::Tensor operator()(torch::Tensor tensor) {
		torch::Tensor X_std = (tensor - data_min) / (data_max - data_min);
		return X_std * (max - min) + min;
	}
	torch::Tensor inverse_transform(torch::Tensor tensor) {
		torch::Tensor Y_std = (tensor - min) / (max - min);
		return Y_std * (data_max - data_min) + data_min;
	}
	void parse(istream& is) {
		data_min = torch::ones({ 1,nn_outputs });
		is.get();
		double temp = 0.0;
		for (int i = 0; i < nn_outputs; i++) {
			is >> temp;
			data_min[0][i] = temp;
		}
		is.get();
		is.get();
		data_max = torch::ones({ 1, nn_outputs });
		is.get();
		for (int i = 0; i < nn_outputs; i++) {
			is >> temp;
			data_max[0][i] = temp;
		}
		is.get(); //']'
		is.get(); //'\n'
		is >> min;
		is >> max;
	}
};

//ode function of Van der Pol equation
class BubDyn {
	double mu = 1.5;
public:
	torch::jit::script::Module model; //the neural network
	torch::Tensor inputs; //reused tensor of inputs
	std_scaler in_transf;
	std_scaler out_transf;

	std::vector<value_type> C; //constants of the right hand side
	Eigen::Matrix<value_type, N / 2, 1> y; //collocation points (half)
	Eigen::Matrix<value_type, N / 2, 1> y_sq; //same, every entry squared
	matrix_type D_E; //Derivative matrix for even functions
	matrix_type D_O; //Derivative matrix for odd functions

	BubDyn() {
		//----------------------------------------------------------
		//bubblemodel initializations
		//----------------------------------------------------------
		const double omega = 2 * M_PI * f;
		value_type pi2wRE = 2 * M_PI / (omega * R_E);

		//constants
		C = std::vector<value_type>(13);
		C[0] = omega * R_E / (2 * M_PI * c_L);
		C[1] = 4 * mu_L / (c_L * rho_L * R_E);
		C[2] = 4 * mu_L / (rho_L * R_E) * pi2wRE;
		C[3] = 2 * sigma * pi2wRE * pi2wRE / (rho_L * R_E);
		C[4] = p_inf / rho_L * pi2wRE * pi2wRE;
		C[5] = p_A / rho_L * pi2wRE * pi2wRE;
		C[6] = pi2wRE * p_inf / (c_L * rho_L);
		C[7] = pi2wRE * p_A / (c_L * rho_L);
		C[8] = 2 * M_PI * pi2wRE * p_A / (c_L * rho_L);
		C[9] = lambda * (gamma - 1) / gamma * pi2wRE / R_E * T_inf / p_inf;
		C[10] = lambda * (gamma - 1) * pi2wRE / R_E * T_inf / p_inf;
		C[11] = (gamma - 1) / gamma;
		C[12] = 1.0 / (3.0 * gamma);

		//Derivative matrices
		Eigen::Matrix<value_type, N, 1> y_full(N);
		value_type rec_cpn = 1.0 / (N - 1);
		for (int i = 0; i < N; i++) {
			y_full[i] = cos(M_PI * i * rec_cpn);
			//std::cout << y_full[i] << std::endl;
		}
		Eigen::Matrix<value_type, N, N> D(N, N);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				if (i == j) {
					if (i == N - 1) {
						D(N - 1, N - 1) = -(1 + 2 * (N - 1) * (N - 1)) / 6.0;
					}
					else if (i == 0) {
						D(0, 0) = (1 + 2 * (N - 1) * (N - 1)) / 6.0;
					}
					else {
						D(i, i) = -y_full[i] / (2.0 * (1.0 - y_full[i] * y_full[i]));
					}
				}
				else {
					D(i, j) = std::pow(-1, i + j) * (i == 0 || i == N - 1 ? 2.0 : 1.0)
						/ ((j == 0 || j == N - 1 ? 2.0 : 1.0) * (y_full[i] - y_full[j]));
				}
			}
		}
		D_E = matrix_type(N / 2, N / 2);
		D_O = matrix_type(N / 2, N / 2);
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				D_E(i, j) = D(i, j) + D(i, N - 1 - j);
			}
		}
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				D_O(i, j) = D(i, j) - D(i, N - 1 - j);
			}
		}
		y = y_full.head<N / 2>();
		y_sq = y.cwiseProduct(y);

		//---------------------------------------------------------
		//neural network initializations
		//---------------------------------------------------------
		inputs = torch::ones({ 1, nn_inputs }, global_tensor_op);

		try {
			model = torch::jit::load(model_file);
			std::vector<torch::jit::IValue> inp;
			inp.push_back(torch::ones({ 1, nn_inputs }, global_tensor_op));
			std::cout << inp << endl;
			// Execute the model and turn its output into a tensor.
			at::Tensor output = model.forward(inp).toTensor().detach();
			std::cout << output << endl;
		}
		catch (const c10::Error& e) {
			std::cerr << "Error loading the model: " << e.what() << endl;
			exit(-1);
		}
		ifstream in(scaler_file);
		if (!in) {
			std::cerr << "Error loading the scalers." << endl;
			exit(-1);
		}
		out_transf.parse(in, nn_outputs);
		in_transf.parse(in, nn_inputs);
		in.close();
	}

	//Rewrites the errors array with the predicted local truncation errors
	double* local_error(double t, double t_next, const double* x, double* errors) {

		//updating inputs
		inputs[0][0] = t_next - t;

		for (int i = 0; i < nn_inputs - 1; i++) {
			inputs[0][i + 1] = x[i];
		}
		inputs[0][nn_inputs - 1] = sin(2 * M_PI * t);

		//scaling
		torch::Tensor scaled = in_transf(inputs);
		std::vector<torch::jit::IValue> inps;
		inps.push_back(scaled);
		//evaluating
		torch::Tensor loc_trun_err = model.forward(inps).toTensor().detach();
		loc_trun_err = out_transf.inverse_transform(loc_trun_err);

		for (int i = 0; i < 3; i++) {
			errors[i] = loc_trun_err[0][i].item<double>();
		}
		errors[3] = 0.0; //boundary condition
		for (int i = 3; i < nn_outputs; i++) {
			errors[i + 1] = loc_trun_err[0][i].item<double>();
		}
		return errors;
	}

	//ODE function. In the pointer x the values are rewritten with the computed slopes
	void operator()(double t, const double* x, double* dxdt) {
		value_type rec_xR = 1.0 / x[0];
		value_type rec_xp = 1.0 / x[2];

		Eigen::Map<const Eigen::Matrix<value_type, N / 2, 1>> z(x + 3);
		Eigen::Map<Eigen::Matrix<value_type, N / 2, 1>> dzdt(dxdt + 3);

		Eigen::Matrix<value_type, N / 2, 1> De_x = D_E * z; //derivative of z (dimless temperature)
		/*for(int i = 0; i < N/2; i++){
			std::cout << "t" << y_sq[i] << std::endl;
		}*/

		//bubble pressure evolution
		dxdt[2] = 3 * rec_xR * (C[10] * rec_xR * De_x[0] - gamma * x[1] * x[2]);

		//discretized PDE of bubble temperature
		dzdt = De_x.cwiseProduct(x[1] * rec_xR * y - C[9] * rec_xR * rec_xR * rec_xp * De_x //this might show error, but it will not fail at compile time, valid syntax
			+ C[12] * rec_xp * dxdt[2] * y)
			+ C[11] * rec_xp * dxdt[2] * z + C[9] * rec_xp * rec_xR * rec_xR * z
			.cwiseProduct(y_sq.cwiseInverse()).cwiseProduct(D_O * (y_sq.cwiseProduct(De_x)));
		dxdt[3] = 0.0; //Boundary condition

		//Keller-Miksis equation
		dxdt[0] = x[1];
		value_type sin2pit = sin(2 * M_PI * t);
		value_type den = x[0] - C[0] * x[0] * x[1] + C[1];
		value_type nom = 0.5 * C[0] * x[1] * x[1] * x[1] - 1.5 * x[1] * x[1] - C[2] * x[1] * rec_xR - C[3] * rec_xR
			+ C[4] * x[2] - C[4] - C[5] * sin2pit + C[6] * x[1] * x[2] - C[6] * x[1] - C[7] * x[1] * sin2pit
			- C[8] * x[0] * cos(2 * M_PI * t) + C[6] * x[0] * dxdt[2];
		dxdt[1] = nom / den;
	}
};

class ODESolver
{
public:

	ODESolver(int order) :order(order) {};

	bool setInitialCondition(double* conds, double at) {
		begin_t = at;
		init_conds = (double*)malloc(sizeof(double) * order);
		for (int u = 0; u < order; u++) {
			init_conds[u] = conds[u];
		}
		return true;
	}
	void setTimeStep(double dt) {
		delta_t = dt;
	}
	bool setStepNumber(int steps) {
		max_l = steps;
		return true;
	}

	void solve(BubDyn& sys, ostream& os) {

		double* vector = (double*)malloc(sizeof(double) * order);
		for (int u = 0; u < order; u++) {
			vector[u] = init_conds[u];
		}

		//preparations
		double* k = (double*)malloc(sizeof(double) * order);
		double* local_error = (double*)malloc(sizeof(double) * order);
		double t = begin_t;
		int l = 0;

		os << t;
		for (int i = 0; i < order; i++) {
			os << " " << vector[i];
		}
		os << endl;

		#pragma warning(disable:6011)
		while (l < max_l) {

			sys.local_error(t, t + delta_t, vector, local_error);
			sys(t, vector, k);
			for (int j = 0; j < order; j++) {
				vector[j] = vector[j] + delta_t * k[j] + delta_t * delta_t * local_error[j];
				//To change to Euler Method uncomment the following, comment out the previous
				//vector[j] = vector[j] + delta_t * k[j];																 
			}
			l++;
			t += delta_t;
			os << t;
			for (int i = 0; i < order; i++) {
				os << " " << vector[i];
			}
			os << endl;
		}
		free(k);
		free(vector);
	}

	~ODESolver() {
		free(init_conds);
	}
private:
	int type = 0;
	int order = 1;
	double* init_conds = 0;
	double begin_t = 0;
	double delta_t = 0.1;
	int max_l = 10;
	double distance(double a, double b) {
		if (a < b)return b - a;
		else return a - b;
	}
};




int main() {
	global_tensor_op = torch::TensorOptions().dtype(torch::kFloat64);
	cout << "BubbleDynamics with DEM started\n" << setprecision(17) << endl;

	ofstream ofs(file_name);
	if (!ofs.is_open()) {
		cout << "File could not be opened: " << file_name << endl;
		exit(-1);
	}
	ofs.precision(17);
	ofs.flags(ios::scientific);
	cout << "Writing file: " << file_name << endl;

	//initial conditions
	double* x = new double[system_order] {1.0, 0.0, 1.0 + 2.0 * sigma / (R_E * p_inf), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

	double t_start = 0.0;
	cout << "Rarrrr" << endl;
	BubDyn bubi;

	ODESolver solver(system_order);
	solver.setInitialCondition(x, 0.0);
	solver.setTimeStep(2e-5);
	solver.setStepNumber(5/2e-5);

	cout << "Solving..." << endl;
	auto t1 = chrono::high_resolution_clock::now();
	solver.solve(bubi, ofs);
	auto t2 = chrono::high_resolution_clock::now();
	//Not valid measurement of DEM computational time. Just a slight indicator
	cout << "Time (ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << endl;

	ofs.flush();
	ofs.close();

	cout << "Ready" << endl;
	return 0;
}