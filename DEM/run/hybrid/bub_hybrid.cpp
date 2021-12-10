/*Deep Euler implementation of a sonochemical bubble model*/

//#define EIGEN_NO_DEBUG
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
const int nn_outputs = N_z;
const int system_order = 4+N_z;
c10::TensorOptions global_tensor_op;

//Modify these to load the correct model
string file_name = "../simulations/bub_hybrid_test.txt";
string model_file = "../../../training/traced_model_bub0.5_hybrid_e51_2112021549.pt";
string scaler_file = "../../../training/scaler_bub0.5_hybrid_2112021549.psca";

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
	std::vector<torch::jit::IValue> inps; //reused neural network input vector
	torch::Tensor inputs; //reused tensor of inputs
public:
	torch::jit::script::Module model; //the neural network
	std_scaler in_transf;
	std_scaler out_transf;

	std::array<double, 5> stage_times;
	std::vector<value_type> C; //constants of the right hand side
	Eigen::Matrix<value_type, N / 2, 1> y; //collocation points (half)
	Eigen::Matrix<value_type, N / 2, 1> y_sq; //same, every entry squared
	matrix_type D_E; //Derivative matrix for even functions
	matrix_type D_O; //Derivative matrix for odd functions

	BubDyn() {
		inps = std::vector<torch::jit::IValue>(1);
		inputs = torch::ones({ 5, nn_inputs }, global_tensor_op);

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
		C[12] = 1.0 / (3 * gamma);

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
		torch::Tensor inputs = torch::ones({ 1, nn_inputs }, global_tensor_op);

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
	torch::Tensor local_error(double t, double dt, const double * x, const double* z) {
		double sinpi = sin(2 * M_PI * t);

		for (int jj = 0; jj < 5; jj++) {
			inputs[jj][1] = x[0];
			inputs[jj][2] = x[1];
			inputs[jj][3] = x[2];
			inputs[jj][nn_inputs - 1] = sinpi;
			//timestep
			inputs[jj][0] = stage_times[jj] * dt;
			//temperature
			for (int i = 0; i < N_z; i++) {
				inputs[jj][i + 4] = z[i+1];
			}
			//scaling
			inputs.index_put_({jj, torch::indexing::Slice()}, in_transf(inputs.index({ jj, torch::indexing::Slice() })));
		}
		inps[0] = inputs;
		//evaluating
		torch::Tensor loc_trun_err = model.forward(inps).toTensor().detach();
		for (int i = 0; i < 5; i++) {
			loc_trun_err.index_put_({ i, torch::indexing::Slice() }, out_transf.inverse_transform(loc_trun_err.index({ i, torch::indexing::Slice() })));
		}
		return loc_trun_err;
	}
	//ODE function of discretized temperature (z)
	void temperature(double t, const double * x, const double* z, double* dzdt) {
		Eigen::Map<const Eigen::Matrix<value_type, N / 2, 1>> z_vector(z);
		Eigen::Map<Eigen::Matrix<value_type, N / 2, 1>> dzdt_vector(dzdt);

		Eigen::Matrix<value_type, N / 2, 1> De_x = D_E * z_vector; //derivative of z (dimless temperature)
		
		value_type rec_xR = 1.0 / x[0];
		value_type rec_xp = 1.0 / x[2];
		value_type dxdt2 = 3 * rec_xR * (C[10] * rec_xR * De_x[0] - gamma * x[1] * x[2]);

		//discretized PDE of bubble temperature
		dzdt_vector = De_x.cwiseProduct(x[1] * rec_xR * y - C[9] * rec_xR * rec_xR * rec_xp * De_x //this might show error, but it will not fail at compile time, valid syntax
			+ C[12] * rec_xp * dxdt2 * y)
			+ C[11] * rec_xp * dxdt2 * z_vector + C[9] * rec_xp * rec_xR * rec_xR * z_vector
			.cwiseProduct(y_sq.cwiseInverse()).cwiseProduct(D_O * (y_sq.cwiseProduct(De_x)));
		dzdt_vector[0] = 0.0; //Boundary condition

		return;
	}

	//ODE function of bubbledynamics (x). In the pointer x the values are rewritten with the computed slopes
	void operator()(double t, const double* x, const double * z, double* dxdt) {
		Eigen::Map<const Eigen::Matrix<value_type, N / 2, 1>> z_vector(z);
		value_type rec_xR = 1.0 / x[0];
		value_type rec_xp = 1.0 / x[2];

		//bubble pressure evolution
		dxdt[2] = 3 * rec_xR * (C[10] * rec_xR * (D_E * z_vector)[0] - gamma * x[1] * x[2]);

		//Keller-Miksis equation
		dxdt[0] = x[1];
		value_type sin2pit = sin(2 * M_PI * t);
		value_type den = x[0] - C[0] * x[0] * x[1] + C[1];
		value_type num = 0.5 * C[0] * x[1] * x[1] * x[1] - 1.5 * x[1] * x[1] - C[2] * x[1] * rec_xR - C[3] * rec_xR
			+ C[4] * x[2] - C[4] - C[5] * sin2pit + C[6] * x[1] * x[2] - C[6] * x[1] - C[7] * x[1] * sin2pit
			- C[8] * x[0] * cos(2 * M_PI * t) + C[6] * x[0] * dxdt[2];
		dxdt[1] = num / den;
	}
};

class BubbleSolver
{
public:

	BubbleSolver(int temperature_size) :z_size(temperature_size) {};

	bool setInitialConditions(const double* conds_x, const double * conds_z, const double at) {
		begin_t = at;
		x_init = (double*)malloc(sizeof(double) * x_size);
		for (int u = 0; u < x_size; u++) {
			x_init[u] = conds_x[u];
		}
		z_init = (double*)malloc(sizeof(double) * z_size);
		for (int u = 0; u < z_size; u++) {
			z_init[u] = conds_z[u];
		}
		return true;
	}
	void setMaxTime(double max_t) {
		t_max = max_t;
	}
	void setTolerances(double rel, double abs) {
		abs_tol = abs;
		rel_tol = rel;
	}

	void solve(BubDyn& sys, ostream& os) {

		double* x = (double*)malloc(sizeof(double) * x_size);
		for (int u = 0; u < x_size; u++) {
			x[u] = x_init[u];
		}
		double* z = (double*)malloc(sizeof(double) * z_size);
		for (int u = 0; u < z_size; u++) {
			z[u] = z_init[u];
		}

		//preparations
		double* z_stage = (double*)malloc(sizeof(double) * z_size);
		double* z_next = (double*)malloc(sizeof(double) * z_size);
		double* dzdt = (double*)malloc(sizeof(double) * z_size);
		double* x_stage = (double*)malloc(sizeof(double) * x_size);
		double* x_tmp = (double*)malloc(sizeof(double) * x_size);
		double* k1 = (double*)malloc(sizeof(double) * x_size);
		double* k2 = (double*)malloc(sizeof(double) * x_size);
		double* k3 = (double*)malloc(sizeof(double) * x_size);
		double* k4 = (double*)malloc(sizeof(double) * x_size);
		double* k5 = (double*)malloc(sizeof(double) * x_size);
		double t = begin_t;
		int l = 0;
		torch::Tensor errors;
		bool accept = true;
		bool nan_detect = false;
		value_type rel_err = 0.0;
		value_type coeff = 1.0;
		sys.stage_times = { 0.2, 0.3, 0.6, 1.0, 7.0 / 8.0 };
		z_stage[0] = 1.0; //fixed, boundary condition
		z_next[0] = 1.0; //same

		os << t;
		for (int i = 0; i < x_size; i++) {
			os << " " << x[i];
		}
		for (int i = 0; i < z_size; i++) {
			os << " " << z[i];
		}
		os << endl;

		if (t_max - t < delta_t) delta_t = t_max - t;

		while (t < t_max) {

			//DEM for temperature----------------------------------------------
			errors = sys.local_error(t, delta_t, x, z); //neural network
			sys.temperature(t, x, z, dzdt);
			//DOPRI for bubbleradius-------------------------------------------
			sys(t, x, z, k1);

			//k2
			for (int j = 1; j < z_size; j++) {
				z_stage[j] = z[j] + 0.2 * delta_t * dzdt[j] +0.04 * delta_t * delta_t * errors[0][j - 1].item<double>();
			}
			for (int j = 0; j < x_size; j++) {
				x_stage[j] = x[j] + 0.2 * delta_t * k1[j];
			}
			sys(t + 0.2 * delta_t, x_stage, z_stage,  k2);

			//k3
			for (int j = 1; j < z_size; j++) {
				z_stage[j] = z[j] + 0.3 * delta_t * dzdt[j] + 0.09 * delta_t * delta_t * errors[1][j-1].item<double>();
			}
			for (int j = 0; j < x_size; j++) {
				x_stage[j] = x[j] + 3.0 / 40.0 * delta_t * k1[j] + 9.0 / 40.0 * delta_t * k2[j];
			}
			sys(t + 0.3 * delta_t, x_stage, z_stage, k3); //k3

			//k4
			for (int j = 1; j < z_size; j++) {
				z_stage[j] = z[j] + 0.6 * delta_t * dzdt[j] + 0.36 * delta_t * delta_t * errors[2][j-1].item<double>();
			}
			for (int j = 0; j < x_size; j++) {
				x_stage[j] = x[j] + delta_t * (0.3 * k1[j] - 0.9 * k2[j] + 6.0 / 5.0 * k3[j]);
			}
			sys(t + 0.6 * delta_t, x_stage, z_stage, k4);

			//k5
			for (int j = 1; j < z_size; j++) {
				z_next[j] = z[j] + delta_t * dzdt[j] + delta_t * delta_t * errors[3][j-1].item<double>();
				//this is the DEM solution
			}
			for (int j = 0; j < x_size; j++) {
				x_stage[j] = x[j] + delta_t * (-11.0 / 54.0 * k1[j] + 5.0 / 2.0 * k2[j] - 70.0 / 27.0 * k3[j] + 35.0 / 27.0 * k4[j]);
			}
			sys(t + delta_t, x_stage, z_next, k5);

			//k6
			for (int j = 1; j < z_size; j++) {
				z_stage[j] = z[j] + 7.0/8.0 * delta_t * dzdt[j] + 49.0/64.0 * delta_t * delta_t * errors[4][j-1].item<double>();
			}
			for (int j = 0; j < x_size; j++) {
				x_stage[j] = x[j] + delta_t * (1631.0 / 55296.0 * k1[j] + 175.0 / 512.0 * k2[j] + 575.0 / 13824.0 * k3[j] + 44275.0 / 110592.0 * k4[j] + 253.0 / 4096.0 * k5[j]);
			}
			sys(t + 7.0 / 8.0 * delta_t, x_stage, z_stage, k2); //k6

			//solution--------------------------------------------
			for (int j = 0; j < x_size; j++) {
				//Main solution:
				x_tmp[j] = x[j] + delta_t * (37.0 / 378.0 * k1[j] + 250.0 / 621.0 * k3[j] + 125.0 / 594.0 * k4[j] + 512.0 / 1771.0 * k2[j]); //k2=k6
				//main solution end
			}
			//secondary solution
			for (int j = 0; j < x_size; j++) {
				x_stage[j] = x[j] + delta_t * (2825.0 / 27648.0 * k1[j] + 18575.0 / 48384.0 * k3[j] + 13525.0 / 55296.0 * k4[j] + 277.0 / 14336.0 * k5[j] + 0.25 * k2[j]); //k2=k6
			}

			//error control---------------------------------------
			accept = true;
			nan_detect = false;
			rel_err = 0;
			for (int j = 0; j < x_size; j++) {
				if (!std::isfinite(x_tmp[j]) || !isfinite(x_stage[j])) {
					accept = false;
					std::cout << "NaN detected!" << std::endl;
					nan_detect = true;
					break;
				}
				k2[j] = distance(x_tmp[j], x_stage[j]); // local error
				k4[j] = abs_tol + std::fmax(std::fabs(x_tmp[j]), std::fabs(x[j])) * rel_tol; //tolerance
				if (k2[j] > k4[j]) {
					accept = false;
				}
				rel_err = std::fmax(rel_err, k2[j] / k4[j]);
			}
			/*if (std::isfinite(rel_err))
				for (int j = 1; j < z_size; j++) {
					if ( !std::isfinite(z_next[j]) ) {
						accept = false;
						std::cout << "NaN detected!" << std::endl;
						nan_detect = true;
						break;
					}
				}*/

			if (!accept) {
				//1/(q+1) = 1/5 = 0.2;
				coeff = safety_factor * std::pow(1.0 / rel_err, 0.2);
				if (!std::isfinite(coeff) || nan_detect) coeff = 0.1;
				delta_t = coeff * delta_t;
				//std::cout << "Not good\n";
				continue;
				//redo this step
			}

			//save------------------------------------------------
			t += delta_t;
			l++;
			coeff = safety_factor * std::pow(1.0 / rel_err, 0.2);
			if (!std::isfinite(coeff)) coeff = 0.1;
			else if (coeff < 0.1) coeff = 0.1;
			else if (coeff > 5.0) coeff = 5.0;

			if (t + coeff * delta_t > t_max)delta_t = t_max - t;
			else delta_t = coeff * delta_t;

			os << t;
			for (int i = 0; i < x_size; i++) {
				x[i] = x_tmp[i];
				os << " " << x[i];
			}
			for (int i = 0; i < z_size; i++) {
				z[i] = z_next[i];
				os << " " << z[i];
			}
			os << endl;
		}
		free(x); free(z);
		free(x_stage); free(x_tmp);
		free(dzdt); 
		free(z_stage); free(z_next);
		free(k1); free(k2); free(k3); free(k4); free(k5);
	}

	~BubbleSolver() {
		free(x_init);
		free(z_init);
	}
private:
	const int x_size = 3;
	int z_size = 8;
	double* x_init = 0;
	double* z_init = 0;
	double begin_t = 0;
	double delta_t = 1e-4;
	double abs_tol = 1e-6;
	double rel_tol = 1e-6;
	const double safety_factor = 0.8;
	int t_max = 10;
	double distance(double a, double b) {
		if (a < b)return b - a;
		else return a - b;
	}
};




int main() {
	global_tensor_op = torch::TensorOptions().dtype(torch::kFloat64);
	std::cout << "BubbleDynamics with DEM started\n" << setprecision(17) << endl;

	ofstream ofs(file_name);
	if (!ofs.is_open()) {
		std::cout << "File could not be opened: " << file_name << endl;
		exit(-1);
	}
	ofs.precision(17);
	ofs.flags(ios::scientific);
	std::cout << "Writing file: " << file_name << endl;

	//initial conditions
	double* x = new double[3] {1.0, 0.0, 1.0 + 2.0 * sigma / (R_E * p_inf)};
	double* z = new double[N/2] {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	double t_start = 0.0;
	std::cout << "Rarrrr" << endl;
	BubDyn bubi;

	BubbleSolver solver(N/2);
	solver.setInitialConditions(x, z, 0.0);
	solver.setTolerances(1e-8, 1e-8);
	solver.setMaxTime(5.0);

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