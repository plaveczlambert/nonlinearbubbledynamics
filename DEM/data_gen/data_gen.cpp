/* Bubble Dynamics with Chebyshev Spectral Collocation */

#include <iostream>
#include <fstream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <boost/numeric/odeint.hpp>
#include <Eigen/Core>

using namespace std;
using namespace boost::numeric::odeint;

const double rho_L = 9.970639504998557e+02;
const double p_inf = 1.0e+5;
const double sigma = 0.071977583160056;
//const double R_E = 10e-6; //1...10u
const double gamma = 1.33;
const double c_L = 1.497251785455527e+03; // water 25 Celsius
const double mu_L = 8.902125058209557e-04; //25 Celsius
const double lambda = 0.6084; //water 25 Celsius
const double T_inf = 298.15; // 25 Celsius
const int N = 16; //24:25s  32:104s 48:731s
const double tolerance = 1e-13;
const double t_max = 0.5;
const int sample = 25000;

typedef double value_type;
typedef vector<value_type> state_type;
typedef Eigen::Matrix<value_type, N/2, N/2> matrix_type;

using namespace boost::numeric;

//ode function of bubble dynamic
class bubble {
public:
	std::vector<value_type> C; //constants of the right hand side
	Eigen::Matrix<value_type, N/2, 1> y; //collocation points (half)
	Eigen::Matrix<value_type, N/2, 1> y_sq;
	matrix_type D_E; //Derivative matrix for even functions
	matrix_type D_O; //Derivative matrix for odd functions
	Eigen::Matrix<value_type, 1, N/2> D_E0; //first row of even D matrix

	//p_A pressure amplitude, f pressure frequence, N number of collocation points
	bubble(value_type p_A, value_type f, value_type R_E){
		const double omega = 2*M_PI*f;
		value_type pi2wRE = 2*M_PI/(omega*R_E);

		C = std::vector<value_type>(13);
		C[0] = omega*R_E/(2*M_PI*c_L);
		C[1] = 4*mu_L/(c_L*rho_L*R_E);
		C[2] = 4*mu_L/(rho_L*R_E)*pi2wRE;
		C[3] = 2*sigma*pi2wRE*pi2wRE/(rho_L*R_E);
		C[4] = p_inf/rho_L*pi2wRE*pi2wRE;
		C[5] = p_A/rho_L*pi2wRE*pi2wRE;
		C[6] = pi2wRE*p_inf/(c_L*rho_L);
		C[7] = pi2wRE*p_A/(c_L*rho_L);
		C[8] = 2*M_PI*pi2wRE*p_A/(c_L*rho_L);
		C[9] = lambda*(gamma-1)/gamma*pi2wRE/R_E*T_inf/p_inf;
		C[10] = lambda*(gamma-1)*pi2wRE/R_E*T_inf/p_inf;
		C[11] = (gamma-1)/gamma;
		C[12] = 1.0/(3.0*gamma);
		

		Eigen::Matrix<value_type, N, 1> y_full(N);
		value_type rec_cpn =  1.0/(N-1);
		for(int i = 0; i < N; i++){
			y_full[i] = cos(M_PI*i*rec_cpn);
			//std::cout << y_full[i] << std::endl;
		}
		Eigen::Matrix<value_type, N, N> D(N, N);
		for(int i = 0; i < N; i++){
			for(int j = 0; j < N; j++){
				if(i == j){
					if(i == N-1){
						D(N-1,N-1) = -(1+2*(N-1)*(N-1))/6.0;
					}else if(i == 0){
						D(0,0) = (1+2*(N-1)*(N-1))/6.0;
					}else{
						D(i,i) = -y_full[i]/(2.0*(1.0-y_full[i]*y_full[i]));
					}
				}else{
					D(i,j) = std::pow(-1,i+j)*(i==0 || i==N-1?2.0:1.0)
							/((j==0 || j==N-1?2.0:1.0) * (y_full[i]-y_full[j]) );
				}
				//cout << D(i,j) <<" ";
			}
			//cout << endl;
		}
		D_E = matrix_type(N/2,N/2);
		D_O = matrix_type(N/2,N/2);
		for(int i = 0; i < N/2;i++){
		    for(int j = 0; j < N/2; j++){
		    	D_E(i,j) = D(i,j) + D(i,N-1-j);
		    	if(i==0) D_E0(j) = D_E(i,j);
		    	//cout << D_E(i,j) <<" ";
			}
		    //cout << endl;
		}
		for(int i = 0; i < N/2;i++){
			for(int j = 0; j < N/2; j++){
				D_O(i,j) = D(i,j) - D(i,N-1-j);
				//cout << D_O(i,j) <<" ";
		    }
			//cout << endl;
		}
		y = y_full.head<N/2>();
		y_sq = y.cwiseProduct(y);
    }

	void operator()(const state_type &x, state_type &dxdt, const value_type t){
	    value_type rec_xR = 1.0 / x[0];
	    value_type rec_xp = 1.0 / x[2];

	    Eigen::Map<const Eigen::Matrix<value_type, N/2,1>> z(x.data()+3);
	    Eigen::Map<Eigen::Matrix<value_type, N/2,1>> dzdt(dxdt.data()+3);

	    Eigen::Matrix<value_type, N/2, 1> De_x = D_E*z; //derivative of z (dimless temperature)
		/*for(int i = 0; i < N/2; i++){
	    	std::cout << "t" << y_sq[i] << std::endl;
	    }*/

	    //bubble pressure evolution
	    dxdt[2] = 3*rec_xR*(C[10]*rec_xR* De_x[0] - gamma*x[1]*x[2]);

	    //discretized PDE of bubble temperature
	    dzdt = De_x.cwiseProduct( x[1]*rec_xR*y - C[9]*rec_xR*rec_xR*rec_xp* De_x //this might show error, but it will not fail at compile time, valid syntax
	    		+ C[12]*rec_xp*dxdt[2]*y )
	    		+ C[11]*rec_xp*dxdt[2]*z + C[9]*rec_xp*rec_xR*rec_xR* z
				.cwiseProduct(y_sq.cwiseInverse()).cwiseProduct(D_O * (y_sq.cwiseProduct(De_x)));
	    dxdt[3] = 0.0; //Boundary condition

	    //Keller-Miksis equation
	    dxdt[0] = x[1];
	    value_type sin2pit = sin(2*M_PI*t);
	    value_type den = x[0] - C[0]*x[0]*x[1] + C[1];
	    value_type nom = 0.5*C[0] * x[1]*x[1]*x[1] - 1.5* x[1]*x[1] - C[2]*x[1]*rec_xR - C[3]*rec_xR
	    		+ C[4] * x[2] - C[4] - C[5] * sin2pit + C[6] * x[1]*x[2] - C[6] * x[1] - C[7] * x[1]*sin2pit
	            - C[8] * x[0]*cos(2*M_PI*t) + C[6] * x[0]*dxdt[2];
	    dxdt[1] = nom/den;
	}
};


//vector<double> fs = {20e3, 100e3, 500e3, 2e6};
//vector<double> pAs = {0.5e5, 1.0e5, 1.5e5, 2.0e5};
typedef runge_kutta_dopri5< state_type , value_type , state_type , value_type > stepper_type;

int main() {
	cout << "Bubble dynamics runs started\n" << setprecision(17) << endl;
	auto t1 = chrono::high_resolution_clock::now();
	
	double f = 100e3;
	double R_E = 10e-6;
	double p_A = 0.5e5;
	
    std::mt19937 gen;
	gen.seed(42); //reporducibility
    std::uniform_real_distribution<> dis(0.0, 0.5);
	
	vector<double> times(sample);
	for(int jj=0;jj<sample;jj++){
		times[jj] = dis(gen);
	}
	/*sort(times.begin(),times.end());
	auto last = unique(times.begin(),times.end()); //to get rid of duplicates, if any
	times.erase(last, times.end());*/
	int sampled = times.size();

	state_type x(3+N/2);
	state_type dxdt(3+N/2);

	bubble bubi(p_A, f, R_E);
	auto stepper = make_controlled( tolerance , tolerance, stepper_type() );
			
	stringstream ss("");
	ss << "../data/bubble_sim_p" << p_A/1e5 << "_f" << f/1e3 << "_Re" << R_E*1e6 << "_N" << N << "_t" << t_max << "_" << sampled << ".txt";
	string file_name = ss.str();
	ofstream ofs(file_name);

	if(!ofs.is_open()){
		cout << "File cannot be opened." << endl;
		exit(-1);
	}
	ofs.precision(17);
	ofs.flags(ios::scientific);
		
	x[0] = 1.0;
	x[1] = 0.0;
	x[2] = 1.0 + 2.0*sigma/(p_inf*R_E);
	for(int i=0; i < N/2;i++) x[3+i] = 1.0;

	double t_start = 0.0;
	
	try{
		for(int jj=0; jj < sampled; jj++){
			integrate_adaptive(boost::ref(stepper), boost::ref(bubi), x, t_start, times[jj], 0.01);
			t_start = times[jj];
			ofs << times[jj] << ", " << x[0];
			for(int i =1; i < 3+N/2;i++){
				ofs << ", " << x[i];
			}
			bubi(x, dxdt, times[jj]);
			for(int i =0; i < 3+N/2;i++){
				ofs << ", " << dxdt[i];
			}
			ofs << ", " << sin(2*M_PI*times[jj]);
			ofs << endl;
		}
		cout << "Written " << sampled << " lines." << endl;
	}catch(int i){
		cout << " - aborted: R_E=" << R_E << ", f=" << f << ", p_A=" << p_A << endl;
	}

	auto t2 = chrono::high_resolution_clock::now();
	cout << "Fertig" << endl;
	cout << "Time (ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << endl;
	return 0;
}
