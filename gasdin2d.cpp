#define _USE_MATH_DEFINES

#include "CSR.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <stdarg.h>
#include <float.h>
#include "gasdin2d.h"
#include "time.h"
#include <amgx_c.h>
#include <amgx_config.h>

struct Point
{
	float x, y, z;
};

enum DATA_LAYER
{
	OLD = 0,
	NEW = 1
};

#define Vector Point
#define _min_ min
#define _max_ max

const char		TASK_NAME[50] = "test";

float	EPS = 1.0e-12;

//	Memory for CUDA threads

const float	TAU = 1.e-4;
const float	TMAX = 0.2; //0.2;

const int		SAVE_STEP = 100;

const float	LIMITER_ALFA = 1.5;
const float	MM_M = 1.;
const float	MM_DX = 1. / 160.;


//! ïàðàìåòðû ñåòêè
const int NX = 2;
const int NY = 2;

// ÷èñëî óçëîâ(è ðåáåð) â îäíîé ÿ÷åéêå.
const int countNodesInCell = 3;

const float XMIN = 0.0;	//0.0
const float XMAX = 1.0;	//1.0
const float YMIN = 0.0;	//0.0
const float YMAX = 1.0;	//0.125

const float HX = (XMAX - XMIN) / NX;
const float HY = (YMAX - YMIN) / NY;

//! óçëû
int			nodesCount;
Point	*	nodes;

//! ÿ÷åéêè
int			cellsCount;
int		**	cellNodes;
int		**	cellNeigh;
int		**	cellEdges;
int		*	cellType;
float	*	cellS;
//Point	*	cellC;
float  *   cellCx;
float  *   cellCy;
float  *   cellCz;
bool	*	cellIsBound;

//! ðåáðà
int			edgesCount;
int		*	edgeNode1;
int		*	edgeNode2;
Vector	*	edgeNormal;
float	*	edgeL;
int		*	edgeCell1;
int		*	edgeCell2;
int		*	edgeType;
Point	*	edgeC;

//! Êîýôôèöèåíòû ìàñøòàáèðîâàíèÿ
//Vector	*	cellDX;
float		*	cellDXx;
float		*	cellDXy;
float		*	cellDXz;

//! ìàññèâû ïåðåìåííûõ
float		*u, *wx, *wy, *wz;
float		*uOld, *wxOld, *wyOld, *wzOld;
float		*fU, *fWx, *fWy, *fWz;
float		*fUOld, *fWxOld, *fWyOld, *fWzOld;

float		resU, resWx, resWy, resWz;
//float		**intU, **intWx, **intWy, **intFi;
float	    *intU, *intWx, *intWy, *intWz;
float      **intFi;					//!< èíòåãðàëû â ïðàâîé ÷àñòè

int funcCount = 3;
float		*cellA, *cellInvA;
float		*matrA, *matrInvA;								//!< ìàòðèöà ìàññ
															//Point		**edgeGP, **cellGP;									//!< óçëû êâàäðàòóð Ãàóññà
															//Point		**edgeGP, *cellGP;
Point		**edgeGP;
float		*cellGPx;
float		*cellGPy;
float		*cellGPz;
float		*edgeWGP, *edgeJ;	//!< âåñà è êîýôôèöèåíòû ïðåîáðàçîâàíèÿ äëÿ êâàäðàòóð Ãàóññà
float		*cellWGP, *cellJ;
int			edgeGPCount, cellGPCount;

FILE	*	fileLog;

int step;
float t;




// begin structures for neyavniy methods

float	*b;
int A_size;
const int A_block_size = 9;
const int A_small_block_size = 3;

// end

void initMatrix(CSRMatrix& A);
void loadGrid(char* fileName);	//!<	×òåíèå ñåòêè èç ôàéëîâ.
void initGrid();				//!<	Ïîñòðîåíèå ñåòêè.
void destroyGrid();				//!<	Îñâîáîæäåíèå ïàìÿòè, îòâåäåííîé äëÿ õðàíåíèÿ ñåòêè.
void initData();				//!<	Âûäåëåíèå ïàìÿòè è çàäàíèå íà÷. óñëîâèé.
void destroyData();				//!<	Îñâîáîæäåíèå ïàìÿòè.
void calcIntegralSurf();		//!<	Âû÷èñëåíèå ïîâåðõíîñòíûõ èíòåãðàëîâ â íåâÿçêå.
void calcIntegralVol();			//!<	Âû÷èñëåíèå îáúåìíûõ èíòåãðàëîâ â íåâÿçêå.
void calcNewValues();			//!<	Âû÷èñëåíèå çíà÷åíèé íà ñëåä. âðåìåííîì øàãå.
void calcResidual();			//!<	Âû÷èñëåíèå íåâÿçîê
void calcGradientsVolIntegral();
void calcGradientsSurfIntegral();
void calcNewGradientValues();
void save(int);					//!<	Ñîõðàíåíèå òåêóùèõ çíà÷åíèé â ôàéë.
void writeVTK(char*, int);
void writeVTK_AMGX(char*, int, float *currentLayerSolution);

float	baseF(int, int, float, float);
float	baseDFDx(int, int, float, float);
float	baseDFDy(int, int, float, float);
void	getFields(float&, float&, float&, int, float, float, DATA_LAYER);
void getFlux(float& FU, float& FWx, float& FWy,
	float Ul, float Wxl, float Wyl,
	float Ur, float Wxr, float Wyr, Vector n, bool isBoundary);
void calcMatr(int iCell);
void calcCellGP(int iCell);
void calcEdgeGP(int iEdge);
void cellAddNeigh(int c1, int c2);
void GetK(float&, float&, float&, float&);
void copyToOld();
void copyToOld(float *lastLayerSolution, float *currentLayerSolution, int N);
void printU();
void initRightPart(float *lastLayerSolution, float *rightPart_data);
void inverseMatr(float* a_src, float *am, int N);
void inverseMatr__(float** a_src, float **am, int N);
void OutPut(const char* str, ...);
int __getEdgeByCells(int c1, int c2);

/*!
Äîïîëíèòåëüíûå ôóíêöèè
*/
float max(float, float);
float min(float, float);
float max(float, float, float);
float min(float, float, float);
inline float zero(float a) { return (abs(a) < EPS) ? 0.0 : a; }
inline float pow_2(float x) { return x*x; }
inline float pow_3(float x) { return x*x*x; }
inline float pow_4(float x) { return x*x*x*x; }

int main()
{
#ifdef _DEBUG
	_controlfp(~(_MCW_EM & (~_EM_INEXACT) & (~_EM_UNDERFLOW)), _MCW_EM);
#endif

	srand(time(0));

	fileLog = fopen("galerkin2d.log", "w");

	//loadGrid("square.1");

	initGrid();

	initData();

	// init base structures
	float *b_data = new float[A_block_size * cellsCount];
	float *lastLayerSolution = new float[A_block_size * cellsCount];
	float *currentLayerSolution = new float[A_block_size * cellsCount];

	memset(b_data, 0.0, sizeof(float));
    memset(lastLayerSolution, 0.0, sizeof(float));	// check info
	memset(currentLayerSolution, 0.0, sizeof(float));

	for (int iCell = 0; iCell < cellsCount; iCell++) {
        lastLayerSolution[A_block_size * iCell + 6] = sin(cellCx[iCell] * M_PI) * sin(cellCy[iCell] * M_PI);
	}

	// begin AMGX init
	AMGX_initialize();
	AMGX_initialize_plugins();

	AMGX_config_handle config;
    AMGX_config_create_from_file(&config, "configs/V.json");

    AMGX_resources_handle rsrc;
    AMGX_resources_create_simple(&rsrc, config);

    AMGX_solver_handle solver;
    AMGX_matrix_handle A_amgx;
    AMGX_vector_handle b_amgx;
    AMGX_vector_handle solution_amgx;

    AMGX_solver_create(&solver, rsrc, AMGX_mode_dFFI, config);
    AMGX_matrix_create(&A_amgx, rsrc, AMGX_mode_dFFI);
    AMGX_vector_create(&b_amgx, rsrc, AMGX_mode_dFFI);
    AMGX_vector_create(&solution_amgx, rsrc, AMGX_mode_dFFI);
	// end AMGX init

	CSRMatrix matrix(A_block_size * cellsCount);

	initRightPart(lastLayerSolution, b_data);

	initMatrix(matrix);

	matrix.printToFile("resMatrix.txt");

	int n_amgx = A_block_size * cellsCount;
	int nnz_amgx = A_small_block_size * A_small_block_size * 3 * cellsCount;

    AMGX_matrix_upload_all(A_amgx, n_amgx, nnz_amgx, 1, 1, matrix.ia, matrix.ja, matrix.a, 0);
	AMGX_vector_upload(b_amgx, n_amgx, 1, b_data);
    AMGX_vector_set_zero(solution_amgx, n_amgx, 1);
	AMGX_solver_setup(solver, A_amgx);

    AMGX_pin_memory(currentLayerSolution, n_amgx * sizeof(float));  // only here???

	//AMGX_write_system(A_amgx, b_amgx, solution_amgx, "static_system.mtx");

	//matrix.printToFile("resMatrix.txt");

	t = 0.0;
	step = 0;

	//copyToOld();
	//writeVTK((char*)TASK_NAME, 0);

    writeVTK_AMGX((char*)TASK_NAME, step, lastLayerSolution);

	while (t < TMAX) {
		t += TAU;
		step++;

		// ************ begin AMGX solver ************

		AMGX_solver_solve_with_0_initial_guess(solver, b_amgx, solution_amgx);

		//AMGX_write_system(A_amgx, b_amgx, solution_amgx, "system_1_step.mtx");

		AMGX_vector_download(solution_amgx, currentLayerSolution);

		copyToOld(lastLayerSolution, currentLayerSolution, n_amgx);

		initRightPart(lastLayerSolution, b_data);

		AMGX_vector_upload(b_amgx, n_amgx, 1, b_data);

        if (step % SAVE_STEP == 0)
        {
            writeVTK_AMGX((char*)TASK_NAME, step, currentLayerSolution);//save(step);
        }
        if (step % 100 == 0)
        {
            OutPut("%8d | time: %15.8e | U: %15.8e | WX: %15.8e | WY: %15.8e |\n", step, t, resU, resWx, resWy);
        }
        if(step == 700)
        {
            printf("runtime = %f", (clock() / 1000.0));
        }

		memset(currentLayerSolution, 0.0, sizeof(float));
        AMGX_vector_set_zero(solution_amgx, n_amgx, 1);

		// ************ end AMGX solver ************


		/*memset(intU, 0, sizeof(float) * cellsCount * funcCount);
		memset(intWx, 0, sizeof(float) * cellsCount * funcCount);
		memset(intWy, 0, sizeof(float) * cellsCount * funcCount);

		calcGradientsVolIntegral();

		calcGradientsSurfIntegral();

		calcNewGradientValues();

		calcIntegralVol();

		calcIntegralSurf();

		calcNewValues();

		printU();

		copyToOld();

		if (step % SAVE_STEP == 0)
		{
			writeVTK((char*)TASK_NAME, step);//save(step);
		}
		if (step % 100 == 0)
		{
			OutPut("%8d | time: %15.8e | U: %15.8e | WX: %15.8e | WY: %15.8e |\n", step, t, resU, resWx, resWy);
		}
		if(step == 700)
		{
			printf("runtime = %f", (clock() / 1000.0));
		}*/
	}

	
	//system("pause");

	destroyGrid();

	fclose(fileLog);

	AMGX_solver_destroy(solver);
	AMGX_vector_destroy(b_amgx);
	AMGX_vector_destroy(solution_amgx);
	AMGX_matrix_destroy(A_amgx);
	AMGX_resources_destroy(rsrc);

	AMGX_finalize_plugins();
	AMGX_finalize();

	return 0;
}

void copyToOld(float *lastLayerSolution, float *currentLayerSolution, int N) {
	for (int i = 0; i < N; i++) {
		lastLayerSolution[i] = currentLayerSolution[i];
	}
}

void printU() {
	FILE * fp = fopen("u_cpu_res", "w");
	for (int i = 0; i < cellsCount; i++) {
		for (int j = 0; j < funcCount; j++) {
			fprintf(fp, "%f ", fU[i * funcCount + j]);
		}
		fprintf(fp, "\n\n");
	}

	fclose(fp);
}

void GetK(float& kxx, float& kxy, float& kyx, float& kyy)
{
	kxx = 1.0;
	kyy = 1.0;
	kxy = 0.0;
	kyx = 0.0;
}


// ***************************************************** Ðàáîòà ñ ñåòêîé ****************************************************************************

void initGrid()
{
	OutPut("Init grid...\n");
	nodesCount = (NX + 1) * (NY + 1);
	nodes = (Point*)malloc(nodesCount * sizeof(Point));

	int k = 0;
	for (int j = 0; j <= NY; j++)
	{ 
		float y = YMIN + HY * j;
		for (int i = 0; i <= NX; i++)
		{
			float x = XMIN + HX * i;
			nodes[k].x = x;
			nodes[k].y = y;
			k++;
		}
	}

	cellsCount = NX * NY * 2;
	cellS = (float*)malloc(cellsCount * sizeof(float));
	cellIsBound = (bool*)malloc(cellsCount * sizeof(bool));
	cellCx = (float*)malloc(cellsCount * sizeof(float));
	cellCy = (float*)malloc(cellsCount * sizeof(float));

	cellNodes = (int**)malloc(cellsCount * sizeof(int*));
	cellEdges = (int**)malloc(cellsCount * sizeof(int*));
	cellNeigh = (int**)malloc(cellsCount * sizeof(int*));

	for (int i = 0; i < cellsCount; i++)
	{
		cellNodes[i] = (int*)malloc(3 * sizeof(int));
		cellEdges[i] = (int*)malloc(3 * sizeof(int));
		cellNeigh[i] = (int*)malloc(3 * sizeof(int));
		cellNeigh[i][0] = -1;
		cellNeigh[i][1] = -1;
		cellNeigh[i][2] = -1;
	}

	edgesCount = NX * NY * 3 + NX + NY;
	edgeNode1 = (int*)malloc(edgesCount * sizeof(int));
	edgeNode2 = (int*)malloc(edgesCount * sizeof(int));
	edgeCell1 = (int*)malloc(edgesCount * sizeof(int));
	edgeCell2 = (int*)malloc(edgesCount * sizeof(int));
	edgeNormal = (Vector*)malloc(edgesCount * sizeof(Vector));
	edgeType = (int*)malloc(edgesCount * sizeof(int));
	edgeC = (Point*)malloc(edgesCount * sizeof(Point));
	edgeL = (float*)malloc(edgesCount * sizeof(float));

	int iCell = 0;
	int iEdge = 0;
	for (int j = 0; j < NY; j++)
	{
		for (int i = 0; i < NX; i++)
		{
			cellNodes[iCell][0] = i + (NX + 1)*j;
			cellNodes[iCell][1] = i + 1 + (NX + 1)*j;
			cellNodes[iCell][2] = i + (NX + 1)*(j + 1);

			cellCx[iCell] = (nodes[cellNodes[iCell][0]].x + nodes[cellNodes[iCell][1]].x + nodes[cellNodes[iCell][2]].x) / 3.0;
			cellCy[iCell] = (nodes[cellNodes[iCell][0]].y + nodes[cellNodes[iCell][1]].y + nodes[cellNodes[iCell][2]].y) / 3.0;
			cellS[iCell] = abs((nodes[cellNodes[iCell][0]].x - nodes[cellNodes[iCell][2]].x) * (nodes[cellNodes[iCell][1]].y - nodes[cellNodes[iCell][2]].y)
				- (nodes[cellNodes[iCell][1]].x - nodes[cellNodes[iCell][2]].x)*(nodes[cellNodes[iCell][0]].y - nodes[cellNodes[iCell][2]].y));

			edgeNode1[iEdge] = cellNodes[iCell][0];
			edgeNode2[iEdge] = cellNodes[iCell][1];
			edgeC[iEdge].x = (nodes[edgeNode1[iEdge]].x + nodes[edgeNode2[iEdge]].x) / 2.0;
			edgeC[iEdge].y = (nodes[edgeNode1[iEdge]].y + nodes[edgeNode2[iEdge]].y) / 2.0;
			edgeCell1[iEdge] = iCell;
			edgeCell2[iEdge] = iCell - NX * 2 + 1;
			edgeNormal[iEdge].x = nodes[edgeNode2[iEdge]].y - nodes[edgeNode1[iEdge]].y;
			edgeNormal[iEdge].y = nodes[edgeNode1[iEdge]].x - nodes[edgeNode2[iEdge]].x;		// warning, x <--> y
			edgeL[iEdge] = sqrt(edgeNormal[iEdge].x*edgeNormal[iEdge].x + edgeNormal[iEdge].y*edgeNormal[iEdge].y);
			edgeNormal[iEdge].x /= edgeL[iEdge];
			edgeNormal[iEdge].y /= edgeL[iEdge];
			cellAddNeigh(edgeCell1[iEdge], edgeCell2[iEdge]);

			if (j == 0)
			{
				edgeType[iEdge] = 1;
				cellIsBound[iCell] = true;
			}
			else {
				edgeType[iEdge] = 0;
				cellIsBound[iCell] = false;
			}
			iEdge++;

			edgeNode1[iEdge] = cellNodes[iCell][1];
			edgeNode2[iEdge] = cellNodes[iCell][2];
			edgeC[iEdge].x = (nodes[edgeNode1[iEdge]].x + nodes[edgeNode2[iEdge]].x) / 2.0;
			edgeC[iEdge].y = (nodes[edgeNode1[iEdge]].y + nodes[edgeNode2[iEdge]].y) / 2.0;
			edgeCell1[iEdge] = iCell;
			edgeCell2[iEdge] = iCell + 1;
			edgeNormal[iEdge].x = nodes[edgeNode2[iEdge]].y - nodes[edgeNode1[iEdge]].y;
			edgeNormal[iEdge].y = nodes[edgeNode1[iEdge]].x - nodes[edgeNode2[iEdge]].x;
			edgeL[iEdge] = sqrt(edgeNormal[iEdge].x*edgeNormal[iEdge].x + edgeNormal[iEdge].y*edgeNormal[iEdge].y);
			edgeNormal[iEdge].x /= edgeL[iEdge];
			edgeNormal[iEdge].y /= edgeL[iEdge];
			cellAddNeigh(edgeCell1[iEdge], edgeCell2[iEdge]);
			edgeType[iEdge] = 0;
			cellIsBound[iCell] = false;
			iEdge++;

			edgeNode1[iEdge] = cellNodes[iCell][2];
			edgeNode2[iEdge] = cellNodes[iCell][0];
			edgeC[iEdge].x = (nodes[edgeNode1[iEdge]].x + nodes[edgeNode2[iEdge]].x) / 2.0;
			edgeC[iEdge].y = (nodes[edgeNode1[iEdge]].y + nodes[edgeNode2[iEdge]].y) / 2.0;
			edgeCell1[iEdge] = iCell;
			edgeCell2[iEdge] = (i == 0) ? -1 : iCell - 1;
			edgeNormal[iEdge].x = nodes[edgeNode2[iEdge]].y - nodes[edgeNode1[iEdge]].y;
			edgeNormal[iEdge].y = nodes[edgeNode1[iEdge]].x - nodes[edgeNode2[iEdge]].x;
			edgeL[iEdge] = sqrt(edgeNormal[iEdge].x*edgeNormal[iEdge].x + edgeNormal[iEdge].y*edgeNormal[iEdge].y);
			edgeNormal[iEdge].x /= edgeL[iEdge];
			edgeNormal[iEdge].y /= edgeL[iEdge];
			cellAddNeigh(edgeCell1[iEdge], edgeCell2[iEdge]);
			if (i == 0)
			{
				edgeType[iEdge] = 4;
				cellIsBound[iCell] = true;
			}
			else {
				edgeType[iEdge] = 0;
				cellIsBound[iCell] = cellIsBound[iCell] || false;
			}
			iEdge++;

			iCell++;

			cellNodes[iCell][0] = i + 1 + (NX + 1)*j;
			cellNodes[iCell][1] = i + 1 + (NX + 1)*(j + 1);
			cellNodes[iCell][2] = i + (NX + 1)*(j + 1);

			cellCx[iCell] = (nodes[cellNodes[iCell][0]].x + nodes[cellNodes[iCell][1]].x + nodes[cellNodes[iCell][2]].x) / 3.0;
			cellCy[iCell] = (nodes[cellNodes[iCell][0]].y + nodes[cellNodes[iCell][1]].y + nodes[cellNodes[iCell][2]].y) / 3.0;
			cellS[iCell] = abs((nodes[cellNodes[iCell][0]].x - nodes[cellNodes[iCell][2]].x)*(nodes[cellNodes[iCell][1]].y - nodes[cellNodes[iCell][2]].y)
				- (nodes[cellNodes[iCell][1]].x - nodes[cellNodes[iCell][2]].x)*(nodes[cellNodes[iCell][0]].y - nodes[cellNodes[iCell][2]].y));
			iCell++;
		}
		iCell--;
		edgeNode1[iEdge] = cellNodes[iCell][0];
		edgeNode2[iEdge] = cellNodes[iCell][1];
		edgeC[iEdge].x = (nodes[edgeNode1[iEdge]].x + nodes[edgeNode2[iEdge]].x) / 2.0;
		edgeC[iEdge].y = (nodes[edgeNode1[iEdge]].y + nodes[edgeNode2[iEdge]].y) / 2.0;
		edgeCell1[iEdge] = iCell;
		edgeCell2[iEdge] = -1;
		edgeNormal[iEdge].x = nodes[edgeNode2[iEdge]].y - nodes[edgeNode1[iEdge]].y;
		edgeNormal[iEdge].y = nodes[edgeNode1[iEdge]].x - nodes[edgeNode2[iEdge]].x;
		edgeL[iEdge] = sqrt(edgeNormal[iEdge].x*edgeNormal[iEdge].x + edgeNormal[iEdge].y*edgeNormal[iEdge].y);
		edgeNormal[iEdge].x /= edgeL[iEdge];
		edgeNormal[iEdge].y /= edgeL[iEdge];
		cellAddNeigh(edgeCell1[iEdge], edgeCell2[iEdge]);
		edgeType[iEdge] = 2;
		cellIsBound[iCell] = true;
		iEdge++;

		iCell++;
	}

	for (int i = 0; i < NX; i++)
	{
		iCell--;
		edgeNode1[iEdge] = cellNodes[iCell][1];
		edgeNode2[iEdge] = cellNodes[iCell][2];
		edgeC[iEdge].x = (nodes[edgeNode1[iEdge]].x + nodes[edgeNode2[iEdge]].x) / 2.0;
		edgeC[iEdge].y = (nodes[edgeNode1[iEdge]].y + nodes[edgeNode2[iEdge]].y) / 2.0;
		edgeCell1[iEdge] = iCell;
		edgeCell2[iEdge] = -1;
		edgeNormal[iEdge].x = nodes[edgeNode2[iEdge]].y - nodes[edgeNode1[iEdge]].y;
		edgeNormal[iEdge].y = nodes[edgeNode1[iEdge]].x - nodes[edgeNode2[iEdge]].x;
		edgeL[iEdge] = sqrt(edgeNormal[iEdge].x*edgeNormal[iEdge].x + edgeNormal[iEdge].y*edgeNormal[iEdge].y);
		edgeNormal[iEdge].x /= edgeL[iEdge];
		edgeNormal[iEdge].y /= edgeL[iEdge];
		cellAddNeigh(edgeCell1[iEdge], edgeCell2[iEdge]);
		edgeType[iEdge] = 3;
		cellIsBound[iCell] = true;
		iEdge++;
		iCell--;
	}

	for (int i = 0; i < edgesCount; i++)
	{
		if (edgeCell2[i] < 0) edgeCell2[i] = -1;
	}

	cellDXx = (float*)malloc(cellsCount * sizeof(float));
	cellDXy = (float*)malloc(cellsCount * sizeof(float));
	for (int i = 0; i < cellsCount; i++)
	{
		cellDXx[i] = max(max(nodes[cellNodes[i][0]].x, nodes[cellNodes[i][1]].x), nodes[cellNodes[i][2]].x) - min(min(nodes[cellNodes[i][0]].x, nodes[cellNodes[i][1]].x), nodes[cellNodes[i][2]].x);
		cellDXy[i] = max(max(nodes[cellNodes[i][0]].y, nodes[cellNodes[i][1]].y), nodes[cellNodes[i][2]].y) - min(min(nodes[cellNodes[i][0]].y, nodes[cellNodes[i][1]].y), nodes[cellNodes[i][2]].y);
	}

	FILE * fp = fopen("grid.gnuplot.txt", "w");

	for (int i = 0; i < cellsCount; i++)
	{
		fprintf(fp, "%f %f \n", nodes[cellNodes[i][0]].x, nodes[cellNodes[i][0]].y);
		fprintf(fp, "%f %f \n", nodes[cellNodes[i][1]].x, nodes[cellNodes[i][1]].y);
		fprintf(fp, "%f %f \n", nodes[cellNodes[i][2]].x, nodes[cellNodes[i][2]].y);
		fprintf(fp, "%f %f \n", nodes[cellNodes[i][0]].x, nodes[cellNodes[i][0]].y);
		fprintf(fp, "\n");
	}

	fclose(fp);
}

void cellAddNeigh(int c1, int c2)
{
	if (c1 < 0 || c2 < 0) return;

	int * neigh = cellNeigh[c1];
	if ((neigh[0] != c2) && (neigh[1] != c2) && (neigh[2] != c2))
	{
		int i = 0;
		while (neigh[i] >= 0) i++;
		neigh[i] = c2;
	}

	neigh = cellNeigh[c2];
	if ((neigh[0] != c1) && (neigh[1] != c1) && (neigh[2] != c1))
	{
		int i = 0;
		while (neigh[i] >= 0) i++;
		neigh[i] = c1;
	}
}

int __getEdgeByCells(int c1, int c2)
{
	for (int iEdge = 0; iEdge < edgesCount; iEdge++)
	{
		if ((edgeCell1[iEdge] == c1 && edgeCell2[iEdge] == c2) || (edgeCell1[iEdge] == c2 && edgeCell2[iEdge] == c1)) return iEdge;
	}
}

int __findEdge(int n1, int n2)
{
	for (int iEdge = 0; iEdge < edgesCount; iEdge++)
	{
		if ((edgeNode1[iEdge] == n1 && edgeNode2[iEdge] == n2) || (edgeNode1[iEdge] == n2 && edgeNode2[iEdge] == n1))
		{
			return iEdge;
		}
	}
	return -1;
}

void loadGrid(char* fName)
{
	char str[50];
	FILE *fp;
	int tmp;

	// ÷èòàåì äàííûå îá ÓÇËÀÕ
	sprintf(str, "%s.node", fName);
	fp = fopen(str, "r");
	fscanf(fp, "%d %d %d %d", &nodesCount, &tmp, &tmp, &tmp);
	nodes = new Point[nodesCount];
	for (int i = 0; i < nodesCount; i++)
	{
		//old fscanf(fp, "%d %lf %lf %d", &tmp, &(nodes[i].x), &(nodes[i].y), &tmp);

		fscanf(fp, "%d %lf %lf %lf", &tmp, &(nodes[i].x), &(nodes[i].y), &(nodes[i].z));
	}
	fclose(fp);

	// ÷èòàåì äàííûå î ß×ÅÉÊÀÕ
	sprintf(str, "%s.ele", fName);
	fp = fopen(str, "r");
	fscanf(fp, "%d %d %d", &cellsCount, &tmp, &tmp);
	cellNodes = new int*[cellsCount];
	cellNeigh = new int*[cellsCount];
	cellEdges = new int*[cellsCount];
	cellS = new float[cellsCount];
	//cellC = new Point[cellsCount];
	cellCx = new float[cellsCount];
	cellCy = new float[cellsCount];
	
	cellCz = new float[cellsCount];

	//cellDX = new Vector[cellsCount];
	cellDXx = new float[cellsCount];
	cellDXy = new float[cellsCount];
	cellDXz = new float[cellsCount];

	cellType = new int[cellsCount];
	for (int i = 0; i < cellsCount; i++)
	{
		cellNodes[i] = new int[3];
		cellNeigh[i] = new int[3];
		cellEdges[i] = new int[3];
	}
	for (int i = 0; i < cellsCount; i++)
	{
		fscanf(fp, "%d %d %d %d %d", &tmp, &cellNodes[i][0], &cellNodes[i][1], &cellNodes[i][2], &cellType[i]);
		cellNodes[i][0]--;
		cellNodes[i][1]--;
		cellNodes[i][2]--;
		//cellC[i].x = (nodes[cellNodes[i][0]].x + nodes[cellNodes[i][1]].x + nodes[cellNodes[i][2]].x) / 3.0;
		//cellC[i].y = (nodes[cellNodes[i][0]].y + nodes[cellNodes[i][1]].y + nodes[cellNodes[i][2]].y) / 3.0;

		cellCx[i] = (nodes[cellNodes[i][0]].x + nodes[cellNodes[i][1]].x + nodes[cellNodes[i][2]].x) / 3.0;
		cellCy[i] = (nodes[cellNodes[i][0]].y + nodes[cellNodes[i][1]].y + nodes[cellNodes[i][2]].y) / 3.0;
		//cellDX[i].x = _max_(fabs(nodes[cellNodes[i][0]].x - nodes[cellNodes[i][1]].x), fabs(nodes[cellNodes[i][1]].x - nodes[cellNodes[i][2]].x), fabs(nodes[cellNodes[i][0]].x - nodes[cellNodes[i][2]].x));
		//cellDX[i].y = _max_(fabs(nodes[cellNodes[i][0]].y - nodes[cellNodes[i][1]].y), fabs(nodes[cellNodes[i][1]].y - nodes[cellNodes[i][2]].y), fabs(nodes[cellNodes[i][0]].y - nodes[cellNodes[i][2]].y));
		cellDXx[i] = _max_(fabs(nodes[cellNodes[i][0]].x - nodes[cellNodes[i][1]].x), fabs(nodes[cellNodes[i][1]].x - nodes[cellNodes[i][2]].x), fabs(nodes[cellNodes[i][0]].x - nodes[cellNodes[i][2]].x));
		cellDXy[i] = _max_(fabs(nodes[cellNodes[i][0]].y - nodes[cellNodes[i][1]].y), fabs(nodes[cellNodes[i][1]].y - nodes[cellNodes[i][2]].y), fabs(nodes[cellNodes[i][0]].y - nodes[cellNodes[i][2]].y));
	}
	fclose(fp);

	// ôîðìèðóåì äàííûå î ÐÅÁÐÀÕ
	sprintf(str, "%s.neigh", fName);
	fp = fopen(str, "r");
	fscanf(fp, "%d %d", &tmp, &tmp);
	//int** neigh;
	//neigh = new int*[cellsCount];
	for (int i = 0; i < cellsCount; i++)
	{
		//neigh[i] = new int[3];
		fscanf(fp, "%d %d %d %d", &tmp, &(cellNeigh[i][0]), &(cellNeigh[i][1]), &(cellNeigh[i][2]));
		cellNeigh[i][0]--;
		cellNeigh[i][1]--;
		cellNeigh[i][2]--;
		/*cellNeigh[i][0] = neigh[i][0];
		cellNeigh[i][1] = neigh[i][1];
		cellNeigh[i][2] = neigh[i][2];*/

		//printf("qweqwe:     %d    %d    %d\n", (cellNeigh[i][0]), (cellNeigh[i][1]), (cellNeigh[i][2]));
	}
	fclose(fp);
	edgesCount = 0;
	for (int i = 0; i < cellsCount; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int p = cellNeigh[i][j];
			if (p > -1)
			{
				for (int k = 0; k < 3; k++)
				{ // óáèðàåì ó ñîñåäà íîìåð ýòîé ÿ÷åéêè, ÷òîáû ãðàíü íå ïîâòîðÿëàñü
					if (cellNeigh[p][k] == i) cellNeigh[p][k] = -1;
				}
				edgesCount++;
			}
			if (p == -2) edgesCount++;
		}
	}

	edgeNode1 = new int[edgesCount];
	edgeNode2 = new int[edgesCount];
	edgeNormal = new Vector[edgesCount];
	edgeL = new float[edgesCount];
	edgeCell1 = new int[edgesCount];
	edgeCell2 = new int[edgesCount];
	edgeType = new int[edgesCount];
	edgeC = new Point[edgesCount];

	int iEdge = 0;
	int * cfi = new int[cellsCount];
	for (int i = 0; i < cellsCount; i++)
	{
		cfi[i] = 0;
	}
	// ::memset(cfi, 0, cCount*sizeof(int));
	for (int i = 0; i < cellsCount; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int p = cellNeigh[i][j];
			if (p != -1)
			{
				edgeNode1[iEdge] = cellNodes[i][(j + 1) % 3];
				edgeNode2[iEdge] = cellNodes[i][(j + 2) % 3];
				edgeC[iEdge].x = (nodes[edgeNode1[iEdge]].x + nodes[edgeNode2[iEdge]].x) / 2.0;
				edgeC[iEdge].y = (nodes[edgeNode1[iEdge]].y + nodes[edgeNode2[iEdge]].y) / 2.0;
				edgeNormal[iEdge].x = nodes[edgeNode2[iEdge]].y - nodes[edgeNode1[iEdge]].y;
				edgeNormal[iEdge].y = nodes[edgeNode1[iEdge]].x - nodes[edgeNode2[iEdge]].x;
				edgeL[iEdge] = sqrt(edgeNormal[iEdge].x*edgeNormal[iEdge].x + edgeNormal[iEdge].y*edgeNormal[iEdge].y);
				edgeNormal[iEdge].x /= edgeL[iEdge];
				edgeNormal[iEdge].y /= edgeL[iEdge];
				edgeCell1[iEdge] = i;
				cellEdges[i][cfi[i]] = iEdge;
				cfi[i]++;
				//edges[iEdge].cnl1 = fabs(edges[iEdge].n.x*(edges[iEdge].c[0].x-cells[edges[iEdge].c1].c.x)+edges[iEdge].n.y*(edges[iEdge].c[0].y-cells[edges[iEdge].c1].c.y) );

				if (p > -1)
				{

					edgeCell2[iEdge] = p;
					cellEdges[p][cfi[p]] = iEdge;
					cfi[p]++;
					//edges[iEdge].cnl2 = fabs(edges[iEdge].n.x*(cells[edges[iEdge].c2].c.x-edges[iEdge].c[0].x)+edges[iEdge].n.y*(cells[edges[iEdge].c2].c.y-edges[iEdge].c[0].y) );
					edgeType[iEdge] = 0;
				}
				if (p == -2)
				{
					edgeCell2[iEdge] = -2;
					//edges[iEdge].cnl2 = 0;
					edgeType[iEdge] = -1;
				}
				iEdge++;
			}
		}
	}

	// ÷òåíèå äàííûõ î ãðàíè÷íûõ ãðàíÿõ	
	sprintf(str, "%s.poly", fName);
	fp = fopen(str, "r");
	int bndCount;
	fscanf(fp, "%d %d %d %d", &tmp, &tmp, &tmp, &tmp);
	fscanf(fp, "%d %d", &bndCount, &tmp);
	for (int i = 0; i < bndCount; i++)
	{
		int n, n1, n2, type;
		fscanf(fp, "%d %d %d %d", &n, &n1, &n2, &type);
		n1--;
		n2--;
		int iEdge = __findEdge(n1, n2);
		if (iEdge >= 0) edgeType[iEdge] = type;
	}
	fclose(fp);

	for (int i = 0; i < cellsCount; i++)
	{
		float a = edgeL[cellEdges[i][0]];
		float b = edgeL[cellEdges[i][1]];
		float c = edgeL[cellEdges[i][2]];
		float p = (a + b + c) / 2.0;
		cellS[i] = sqrt(p*(p - a)*(p - b)*(p - c));
		//cellDX[i].x = edgeL[cellEdges[i][0]] * edgeL[cellEdges[i][1]] * edgeL[cellEdges[i][2]] / (4.0*cellS[i]);
		//cellDX[i].y = cellDX[i].xx;
	}

	//for (int i = 0; i < cellsCount; i++)
	//{
	//	//delete[] neigh[i];
	//}
	//delete[] neigh;
	delete[] cfi;

	//fp = fopen("grid.gnuplot.txt", "w");
	//for (int i = 0; i < cellsCount; i++)
	//{
	//	fprintf(fp, "%f %f \n", nodes[cellNodes[i][0]].x, nodes[cellNodes[i][0]].y);
	//	fprintf(fp, "%f %f \n", nodes[cellNodes[i][1]].x, nodes[cellNodes[i][1]].y);
	//	fprintf(fp, "%f %f \n", nodes[cellNodes[i][2]].x, nodes[cellNodes[i][2]].y);
	//	fprintf(fp, "%f %f \n", nodes[cellNodes[i][0]].x, nodes[cellNodes[i][0]].y);
	//	fprintf(fp, "\n");

	//}
	//fclose(fp);
}

// **************************************************************************************************************************************************

// ****************************************************** Èíèöèàëèçàöèÿ *****************************************************************************
void initData()
{
	OutPut("Init data:\n");
	OutPut("  - allocate memory...\n");

	funcCount = 3;
	edgeGPCount = 2;	// Ìû áóäåì ñ÷èòàòü òîëüêî äëÿ 2 
	cellGPCount = 3;	// Ìû áóäåì ñ÷èòàòü òîëüêî äëÿ 3

	int matrixDim = cellsCount * funcCount;

	u = (float*)malloc(cellsCount * sizeof(float));
	wx = (float*)malloc(cellsCount * sizeof(float));
	wy = (float*)malloc(cellsCount * sizeof(float));

	uOld = (float*)malloc(cellsCount * sizeof(float));
	wxOld = (float*)malloc(cellsCount * sizeof(float));
	wyOld = (float*)malloc(cellsCount * sizeof(float));

	fU = (float*)malloc(cellsCount * funcCount * sizeof(float));
	fWx = (float*)malloc(cellsCount * funcCount * sizeof(float));
	fWy = (float*)malloc(cellsCount * funcCount * sizeof(float));

	fUOld = (float*)malloc(cellsCount * funcCount * sizeof(float));
	fWxOld = (float*)malloc(cellsCount * funcCount * sizeof(float));
	fWyOld = (float*)malloc(cellsCount * funcCount * sizeof(float));

	//***************************************************CUDA INIT VARS **********************************************************************************
	//cudaMalloc((void**)&intU_dev, cellsCount * funcCount * sizeof(float));
	//cudaMalloc((void**)&intWx_dev, cellsCount * funcCount * sizeof(float));
	//cudaMalloc((void**)&intWy_dev, cellsCount * funcCount * sizeof(float));

	//cudaMalloc((void**)&cellGPx_dev, cellsCount * cellGPCount * sizeof(float));
	//cudaMalloc((void**)&cellGPy_dev, cellsCount * cellGPCount * sizeof(float));

	//cudaMalloc((void**)&fU_dev, cellsCount * funcCount * sizeof(float));
	//cudaMalloc((void**)&fWx_dev, cellsCount * funcCount * sizeof(float));
	//cudaMalloc((void**)&fWy_dev, cellsCount * funcCount * sizeof(float));

	//cudaMalloc((void**)&fUOld_dev, cellsCount * funcCount * sizeof(float));
	//cudaMalloc((void**)&fWxOld_dev, cellsCount * funcCount * sizeof(float));
	//cudaMalloc((void**)&fWyOld_dev, cellsCount * funcCount * sizeof(float));
	//***************************************************CUDA INIT VARS **********************************************************************************



	//intU = (float**)malloc(cellsCount * sizeof(float*));
	intU = (float*)malloc(cellsCount * funcCount * sizeof(float));
	intWx = (float*)malloc(cellsCount * funcCount * sizeof(float));
	intWy = (float*)malloc(cellsCount * funcCount * sizeof(float));
	//intWx = (float**)malloc(cellsCount * sizeof(float*));
	//intWy = (float**)malloc(cellsCount * sizeof(float*));
	intFi = (float**)malloc(cellsCount * sizeof(float*));

	matrA = new float[matrixDim * matrixDim];
	matrInvA = new float[matrixDim * matrixDim];

	memset(matrA, 0, sizeof(float) * matrixDim * matrixDim);
	memset(matrInvA, 0, sizeof(float) * matrixDim * matrixDim);

	cellA = new float[funcCount * funcCount];
	cellInvA = new float[funcCount * funcCount];

	//cellGP = (Point**)malloc(cellsCount * sizeof(Point*));
	//cellGP = (Point*)malloc(cellsCount * cellGPCount * sizeof(Point));
	cellGPx = (float*)malloc(cellsCount * cellGPCount * sizeof(float));
	cellGPy = (float*)malloc(cellsCount * cellGPCount * sizeof(float));
	edgeGP = (Point**)malloc(edgesCount * sizeof(Point*));

	//***************************************************CUDA INIT VARS **********************************************************************************
	//cudaMalloc((void**)&cellWGP_dev, cellGPCount * sizeof(float));
	//***************************************************CUDA INIT VARS **********************************************************************************

	cellWGP = (float*)malloc(cellGPCount * sizeof(float));
	edgeWGP = (float*)malloc(edgeGPCount * sizeof(float));

	for (int i = 0; i < cellsCount; i++)
	{
		//intU[i] = (float*)malloc(funcCount * sizeof(float));
		intFi[i] = (float*)malloc(funcCount * sizeof(float));
		//intWx[i] = (float*)malloc(funcCount * sizeof(float));
		//intWy[i] = (float*)malloc(funcCount * sizeof(float));

		//matrA[i] = new float[funcCount * funcCount];
		//matrInvA[i] = new float[funcCount * funcCount];
		/*for (int j = 0; j < funcCount; j++)
		{
		matrA[i][j] = (float*)malloc(funcCount * sizeof(float));
		matrInvA[i][j] = (float*)malloc(funcCount * sizeof(float));
		}*/

		//cellGP[i] = (Point*)malloc(cellGPCount * sizeof(Point));
	}
	for (int i = 0; i < edgesCount; i++)
	{
		edgeGP[i] = (Point*)malloc(edgeGPCount * sizeof(Point));
	}
	cellJ = (float*)malloc(cellsCount * sizeof(float));

	//**************************************************CUDA INIT VARS **********************************************************************************
	//cudaMalloc((void**)&cellJ_dev, cellsCount * sizeof(float));

	//**************************************************CUDA INIT VARS **********************************************************************************
	edgeJ = (float*)malloc(edgesCount * sizeof(float));

	OutPut("  - init start data...\n");
	float x2 = (XMAX + XMIN) / 2.0;

	// ******************************* Íà÷àëüíûå óñëîâèÿ **********************************************

	bool isBoundary = false;

	for (int i = 0; i < cellsCount; i++)
	{
		//for (int j = 0; j < countNodesInCell; j++)
		//{
		//printf("%f,    %f,     %f\n", cellNeigh[i][0], cellNeigh[i][1], cellNeigh[i][2]);

		//if (cellNeigh[i][j] == -1)		//åñëè ÿ÷åéêà ÿâëÿåòñÿ ãðàíè÷íîé
		//{
		//isBoundary = true;
		//}
		//}

		//if (isBoundary)	//â êàæäîé ÿ÷åéêå
		//{
		fU[i * funcCount + 0] = sin(cellCx[i] * M_PI) * sin(cellCy[i] * M_PI);
		fU[i * funcCount + 1] = 0.0;
		fU[i * funcCount + 2] = 0.0;

		fWx[i * funcCount + 0] = 0.0;
		fWy[i * funcCount + 0] = 0.0;
		fWx[i * funcCount + 1] = 0.0;
		fWy[i * funcCount + 1] = 0.0;
		fWx[i * funcCount + 2] = 0.0;
		fWy[i * funcCount + 2] = 0.0;
		//}
		/*else
		{
		for (int j = 0; j < funcCount; j++)
		{
		fU[i][j] = 0.0;
		fWx[i][j] = 0.0;
		fWy[i][j] = 0.0;
		}
		}*/

		//isBoundary = false;
	}

	// ************************************************************************************************
	OutPut("  - init gaussian points...\n");
	for (int i = 0; i < cellsCount; i++)
	{
		calcCellGP(i);
		calcMatr(i);
	}

	/*for(int i = 0; i < matrixDim; i++)
	{
	for (int j = 0; j < matrixDim; j++)
	{
	printf("%f   ", matrA[i * matrixDim + j]);
	}

	printf("\n\n");
	}*/

	for (int i = 0; i < edgesCount; i++)
	{
		calcEdgeGP(i);
	}

	//OutPut("  - init limiters...\n");
	//initLimiterParameters();


	copyToOld();
	//OutPut("\n\n");
	//writeVTK((char*)TASK_NAME, 0); //save(0);
}

// **************************************************************************************************************************************************

// ****************************************************** Áàçèñíûå ôóíêöèè **************************************************************************
float baseF(int fNum, int iCell, float x, float y)
{
	switch (fNum)
	{
	case 0:
		return 1.0;
		break;
	case 1:
		return (x - cellCx[iCell]) / cellDXx[iCell];
		break;
	case 2:
		return (y - cellCy[iCell]) / cellDXy[iCell];
		break;
	case 3:
		return (x - cellCx[iCell])*(x - cellCx[iCell]) / cellDXx[iCell] / cellDXx[iCell];
		break;
	case 4:
		return (y - cellCy[iCell])*(y - cellCy[iCell]) / cellDXy[iCell] / cellDXy[iCell];
		break;
	case 5:
		return (x - cellCx[iCell])*(y - cellCy[iCell]) / cellDXx[iCell] / cellDXy[iCell];
		break;
	default:
		OutPut("ERROR! Wrong number of basic function.\n");
		exit(-1);
	}
}

float baseDFDx(int fNum, int iCell, float x, float y)							//??????????????????????????
{
	switch (fNum)
	{
	case 0:
		return 0.0;
		break;
	case 1:
		return 1.0 / cellDXx[iCell];
		break;
	case 2:
		return 0.0;
		break;
	case 3:
		return ((x - cellCx[iCell]) + (x - cellCx[iCell])) / cellDXx[iCell] / cellDXx[iCell];
		break;
	case 4:
		return 0.0;
		break;
	case 5:
		return (y - cellCy[iCell]) / cellDXx[iCell] / cellDXy[iCell];
		break;
	default:
		OutPut("ERROR! Wrong number of basic function.\n");
		exit(-1);
	}
}

float baseDFDy(int fNum, int iCell, float x, float y)										//???????????????????????
{
	switch (fNum)
	{
	case 0:
		return 0.0;
		break;
	case 1:
		return 0.0;
		break;
	case 2:
		return 1.0 / cellDXy[iCell];
		break;
	case 3:
		return 0.0;
		break;
	case 4:
		return ((y - cellCy[iCell]) + (y - cellCy[iCell])) / cellDXy[iCell] / cellDXy[iCell];
		break;
	case 5:
		return (x - cellCx[iCell]) / cellDXx[iCell] / cellDXy[iCell];
		break;
	default:
		OutPut("ERROR! Wrong number of basic function.\n");
		exit(-1);
	}
}

// **************************************************************************************************************************************************

// *********************************************** Äàííûå äëÿ êâàäðàòóð Ãàóññà **********************************************************************
void calcCellGP(int iCell)	//Ñäåëàë äëÿ 3¸õ òî÷åê.
{
	float a = 1.0 / 6.0;
	float b = 2.0 / 3.0;
	float x1 = nodes[cellNodes[iCell][0]].x;
	float y1 = nodes[cellNodes[iCell][0]].y;
	float x2 = nodes[cellNodes[iCell][1]].x;
	float y2 = nodes[cellNodes[iCell][1]].y;
	float x3 = nodes[cellNodes[iCell][2]].x;
	float y3 = nodes[cellNodes[iCell][2]].y;
	float a1 = x1 - x3;
	float a2 = y1 - y3;
	float b1 = x2 - x3;
	float b2 = y2 - y3;
	float c1 = x3;
	float c2 = y3;

	cellWGP[0] = 1.0 / 6.0;
	//cellGP[iCell][0].x = a1*a + b1*a + c1;
	//cellGP[iCell][0].y = a2*a + b2*a + c2;
	//cellGP[iCell * cellGPCount + 0].x = a1*a + b1*a + c1;
	//cellGP[iCell * cellGPCount + 0].y = a2*a + b2*a + c2;

	cellGPx[iCell * cellGPCount + 0] = a1*a + b1*a + c1;
	cellGPy[iCell * cellGPCount + 0] = a2*a + b2*a + c2;

	cellWGP[1] = 1.0 / 6.0;
	//cellGP[iCell][1].x = a1*a + b1*b + c1;
	//cellGP[iCell][1].y = a2*a + b2*b + c2;
	//cellGP[iCell * cellGPCount + 1].x = a1*a + b1*b + c1;
	//cellGP[iCell * cellGPCount + 1].y = a2*a + b2*b + c2;

	cellGPx[iCell * cellGPCount + 1] = a1*a + b1*b + c1;
	cellGPy[iCell * cellGPCount + 1] = a2*a + b2*b + c2;

	cellWGP[2] = 1.0 / 6.0;
	//cellGP[iCell][2].x = a1*b + b1*a + c1;
	//cellGP[iCell][2].y = a2*b + b2*a + c2;
	//cellGP[iCell * cellGPCount + 2].x = a1*b + b1*a + c1;
	//cellGP[iCell * cellGPCount + 2].y = a2*b + b2*a + c2;

	cellGPx[iCell * cellGPCount + 2] = a1*b + b1*a + c1;
	cellGPy[iCell * cellGPCount + 2] = a2*b + b2*a + c2;

	cellJ[iCell] = a1*b2 - a2*b1;
}

void calcEdgeGP(int iEdge)	//Ñäåëàë äëÿ 2óõ òî÷åê.
{
	float gp1 = -1.0 / sqrt(3.0);
	float gp2 = 1.0 / sqrt(3.0);
	float x1 = nodes[edgeNode1[iEdge]].x;
	float y1 = nodes[edgeNode1[iEdge]].y;
	float x2 = nodes[edgeNode2[iEdge]].x;
	float y2 = nodes[edgeNode2[iEdge]].y;

	edgeWGP[0] = 1.0;
	edgeGP[iEdge][0].x = (x1 + x2) / 2.0 + gp1*(x2 - x1) / 2.0;
	edgeGP[iEdge][0].y = (y1 + y2) / 2.0 + gp1*(y2 - y1) / 2.0;

	edgeWGP[1] = 1.0;
	edgeGP[iEdge][1].x = (x1 + x2) / 2.0 + gp2*(x2 - x1) / 2.0;
	edgeGP[iEdge][1].y = (y1 + y2) / 2.0 + gp2*(y2 - y1) / 2.0;

	edgeJ[iEdge] = sqrt(pow_2(x2 - x1) + pow_2(y2 - y1))*0.5;
}

// **************************************************************************************************************************************************

// ****************************************** Âû÷èñëåíèå èíòåãðàëîâ â ïðàâûõ ÷àñòÿõ *****************************************************************
void calcGradientsVolIntegral()	// îáúåìíûé èíòåãðàë äëÿ wx è wy îò u íà ïðåäûäóùåì âðåìåííîì øàãå.
{
	float intWxtmp, intWytmp;
	float gpU, gpWx, gpWy;

	for (int iCell = 0; iCell < cellsCount; iCell++)
	{
		for (int iF = 0; iF < funcCount; iF++)
		{
			intWxtmp = 0.0;
			intWytmp = 0.0;

			for (int iGP = 0; iGP < cellGPCount; iGP++)
			{
				//getFields(gpU, gpWx, gpWy, iCell, cellGP[iCell][iGP].x, cellGP[iCell][iGP].y, OLD);	// áåðåì layer = OLD ñ ïðîøëîãî âðåìåííîãî øàãà

				//float bfDFDx = baseDFDx(iF, iCell, cellGP[iCell][iGP].x, cellGP[iCell][iGP].y);
				//float bfDFDy = baseDFDy(iF, iCell, cellGP[iCell][iGP].x, cellGP[iCell][iGP].y);
				//getFields(gpU, gpWx, gpWy, iCell, cellGP[iCell * cellGPCount + iGP].x, cellGP[iCell * cellGPCount + iGP].y, OLD);
				getFields(gpU, gpWx, gpWy, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP], OLD);
				//float bfDFDx = baseDFDx(iF, iCell, cellGP[iCell * cellGPCount + iGP].x, cellGP[iCell * cellGPCount + iGP].y);
				//float bfDFDy = baseDFDy(iF, iCell, cellGP[iCell * cellGPCount + iGP].x, cellGP[iCell * cellGPCount + iGP].y);
				float bfDFDx = baseDFDx(iF, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);
				float bfDFDy = baseDFDy(iF, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);

				intWxtmp += cellWGP[iGP] * (gpU * bfDFDx);
				intWytmp += cellWGP[iGP] * (gpU * bfDFDy);
				//printf("%f\n", gpU);
			}

			if (abs(intWxtmp) <= EPS) intWxtmp = 0.0;
			if (abs(intWytmp) <= EPS) intWytmp = 0.0;


			//intWx[iCell][iF] -= intWxtmp * cellJ[iCell];
			intWx[iCell * funcCount + iF] -= intWxtmp * cellJ[iCell];
			//intWy[iCell][iF] -= intWytmp * cellJ[iCell];
			intWy[iCell * funcCount + iF] -= intWytmp * cellJ[iCell];
		}
	}
}

void calcIntegralVol()	// Ñ÷èòàåì îáúåìíûé èíòåãðàë
{
	float intUtmp;
	float gpU, gpWx, gpWy;

	for (int iCell = 0; iCell < cellsCount; iCell++)
	{
		for (int iF = 0; iF < funcCount; iF++) {
			intUtmp = 0.0;

			for (int iGP = 0; iGP < cellGPCount; iGP++)
			{
				//getFields(gpU, gpWx, gpWy, iCell, cellGP[iCell][iGP].x, cellGP[iCell][iGP].y, NEW);
				//getFields(gpU, gpWx, gpWy, iCell, cellGP[iCell * cellGPCount + iGP].x, cellGP[iCell * cellGPCount + iGP].y, NEW);
				getFields(gpU, gpWx, gpWy, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP], NEW);

				//float bfDFDx = baseDFDx(iF, iCell, cellGP[iCell][iGP].x, cellGP[iCell][iGP].y);
				//float bfDFDy = baseDFDy(iF, iCell, cellGP[iCell][iGP].x, cellGP[iCell][iGP].y);
				//float bfDFDx = baseDFDx(iF, iCell, cellGP[iCell * cellGPCount + iGP].x, cellGP[iCell * cellGPCount + iGP].y);
				//float bfDFDy = baseDFDy(iF, iCell, cellGP[iCell * cellGPCount + iGP].x, cellGP[iCell * cellGPCount + iGP].y);
				float bfDFDx = baseDFDx(iF, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);
				float bfDFDy = baseDFDy(iF, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);

				intUtmp += cellWGP[iGP] * (gpWx * bfDFDx + gpWy * bfDFDy);
			}

			if (abs(intUtmp) <= EPS) intUtmp = 0.0;

			//intU[iCell][iF] -= intUtmp * cellJ[iCell];
			intU[iCell * funcCount + iF] -= intUtmp * cellJ[iCell];
		}
	}
}

void calcGradientsSurfIntegral()
{
	float FU, FWx, FWy;	// ïîòîêè íà ãðàíèöå ÿ÷åéêè
							//float FUl, FWxl, FWyl;	// ïîòîêè íà ãðàíèöå "-"
							//float FUr, FWxr, FWyr;  // ïîòîêè íà ãðàíèöå "+"
	float Ul = 0.0, Wxl = 0.0, Wyl = 0.0;  // ïðèì. ïåðåìåííûå â ñîñåäíåé ÿ÷åéêå
	float Ur = 0.0, Wxr = 0.0, Wyr = 0.0;  // ïðèì. ïåðåìåííûå â ñîñåäíåé ÿ÷åéêå



											// äîáàâèë êîýôôèöåíòû - çàãëóøêè
	float kxx, kyy, kxy, kyx;
	GetK(kxx, kxy, kyx, kyy);

	//float alpha;
	int c1, c2;
	bool isBoundary = false;
	for (int i = 0; i < edgesCount; i++)
	{
		c1 = edgeCell1[i];	// âíóòðÿííÿÿ ÿ÷åéêà äëÿ i - îãî ðåáðà
		c2 = edgeCell2[i];	// âíåøíÿÿ ÿ÷åéêà äëÿ i - îãî ðåáðà
		if (c2 >= 0) {
			isBoundary = false;

			for (int iF = 0; iF < funcCount; iF++) // êàæäûé ñëó÷àé ïðè ñêàëÿðíîì óìíîæåíèè óðàâíåíèé íà baseF( iF, ... )
			{
				float tmpIntWx1 = 0.0, tmpIntWy1 = 0.0; // äëÿ ÿ÷åéêè Ñ1
				float tmpIntWx2 = 0.0, tmpIntWy2 = 0.0; // äëÿ ÿ÷åéêè Ñ2
				for (int iGP = 0; iGP < edgeGPCount; iGP++)
				{
					getFields(Ul, Wxl, Wyl, c1, edgeGP[i][iGP].x, edgeGP[i][iGP].y, OLD);	// áåðåì layer = OLD.
					getFields(Ur, Wxr, Wxr, c2, edgeGP[i][iGP].x, edgeGP[i][iGP].y, OLD);	// áåðåì layer = OLD.
					getFlux(FU, FWx, FWy, Ul, Wxl, Wyl, Ur, Wxr, Wyr, edgeNormal[i], isBoundary);		//ïîëó÷àåì çíà÷åíèÿ ïîòîêîâûõ ïåðåìåííûõ

					float cGP1 = edgeWGP[iGP] * baseF(iF, c1, edgeGP[i][iGP].x, edgeGP[i][iGP].y);
					float cGP2 = edgeWGP[iGP] * baseF(iF, c2, edgeGP[i][iGP].x, edgeGP[i][iGP].y);

					// Îñòàâèë ïî îäíîìó èíòåãðàëó ïðè Kxx è Kyy

					tmpIntWx1 += (kxx * FU * cGP1 * edgeNormal[i].x);
					tmpIntWy1 += (kyy * FU * cGP1 * edgeNormal[i].y);

					tmpIntWx2 += (kxx * FU * cGP2 * edgeNormal[i].x);
					tmpIntWy2 += (kyy * FU * cGP2 * edgeNormal[i].y);
				}

				if (abs(tmpIntWx1) <= EPS) tmpIntWx1 = 0.0;
				if (abs(tmpIntWy1) <= EPS) tmpIntWy1 = 0.0;

				//intWx[c1][iF] += tmpIntWx1 * edgeJ[i];
				intWx[c1 * funcCount + iF] += tmpIntWx1 * edgeJ[i];
				//intWy[c1][iF] += tmpIntWy1 * edgeJ[i];
				intWy[c1 * funcCount + iF] += tmpIntWy1 * edgeJ[i];

				if (abs(tmpIntWx2) <= EPS) tmpIntWx2 = 0.0;
				if (abs(tmpIntWy2) <= EPS) tmpIntWy2 = 0.0;

				//intWx[c2][iF] -= tmpIntWx2 * edgeJ[i];
				intWx[c2 * funcCount + iF] -= tmpIntWx2 * edgeJ[i];
				//intWy[c2][iF] -= tmpIntWy2 * edgeJ[i];
				intWy[c2 * funcCount + iF] -= tmpIntWy2 * edgeJ[i];
			}
		}
		else { // C2 < 0, ãðàíè÷íàÿ ñòîðîíà
			isBoundary = true;

			for (int iF = 0; iF < funcCount; iF++) // êàæäûé ñëó÷àé ïðè ñêàëÿðíîì óìíîæåíèè óðàâíåíèé íà baseF( iF, ... )
			{
				float tmpIntWx1 = 0.0, tmpIntWy1 = 0.0; // äëÿ ÿ÷åéêè Ñ1
				for (int iGP = 0; iGP < edgeGPCount; iGP++)
				{
					getFields(Ul, Wxl, Wyl, c1, edgeGP[i][iGP].x, edgeGP[i][iGP].y, OLD);	// áåðåì layer = OLD.

																							// Çäåñü áûëî ìíîãî êîäà...
					float tmp = 0;
					getFlux(FU, FWx, FWy, Ul, Wxl, Wyl, tmp, tmp, tmp, edgeNormal[i], isBoundary);

					float cGP1 = edgeWGP[iGP] * baseF(iF, c1, edgeGP[i][iGP].x, edgeGP[i][iGP].y);

					tmpIntWx1 += (kxx * FU * cGP1 * edgeNormal[i].x);
					tmpIntWy1 += (kyy * FU * cGP1 * edgeNormal[i].y);
				}

				if (abs(tmpIntWx1) <= EPS) tmpIntWx1 = 0.0;
				if (abs(tmpIntWy1) <= EPS) tmpIntWy1 = 0.0;

				//intWx[c1][iF] += tmpIntWx1 * edgeJ[i];
				intWx[c1 * funcCount + iF] += tmpIntWx1 * edgeJ[i];
				//intWy[c1][iF] += tmpIntWy1 * edgeJ[i];
				intWy[c1 * funcCount + iF] += tmpIntWy1 * edgeJ[i];
			}
		}
	}
}

void calcIntegralSurf()	// Ñ÷èòàåì ïîâåðõíîñòíûé èíòåãðàë
{
	float FU, FWx, FWy;	// ïîòîêè íà ãðàíèöå ÿ÷åéêè
							//float FUl, FWxl, FWyl;	// ïîòîêè íà ãðàíèöå "-"
							//float FUr, FWxr, FWyr;  // ïîòîêè íà ãðàíèöå "+"
	float Ul = 0.0, Wxl = 0.0, Wyl = 0.0;  // ïðèì. ïåðåìåííûå â ñîñåäíåé ÿ÷åéêå
	float Ur = 0.0, Wxr = 0.0, Wyr = 0.0;  // ïðèì. ïåðåìåííûå â ñîñåäíåé ÿ÷åéêå

											//float alpha;
	int c1, c2;
	bool isBoundary = false;
	for (int i = 0; i < edgesCount; i++)
	{
		c1 = edgeCell1[i];	// âíóòðÿííÿÿ ÿ÷åéêà äëÿ i - îãî ðåáðà
		c2 = edgeCell2[i];	// âíåøíÿÿ ÿ÷åéêà äëÿ i - îãî ðåáðà
		if (c2 >= 0) {
			isBoundary = false;

			for (int iF = 0; iF < funcCount; iF++) // êàæäûé ñëó÷àé ïðè ñêàëÿðíîì óìíîæåíèè óðàâíåíèé íà baseF( iF, ... )
			{
				float tmpIntU1 = 0.0, tmpIntWx1 = 0.0, tmpIntWy1 = 0.0; // äëÿ ÿ÷åéêè Ñ1
				float tmpIntU2 = 0.0, tmpIntWx2 = 0.0, tmpIntWy2 = 0.0; // äëÿ ÿ÷åéêè Ñ2
				for (int iGP = 0; iGP < edgeGPCount; iGP++)
				{
					getFields(Ul, Wxl, Wyl, c1, edgeGP[i][iGP].x, edgeGP[i][iGP].y, NEW);
					getFields(Ur, Wxr, Wxr, c2, edgeGP[i][iGP].x, edgeGP[i][iGP].y, NEW);
					getFlux(FU, FWx, FWy, Ul, Wxl, Wyl, Ur, Wxr, Wyr, edgeNormal[i], isBoundary);		//ïîëó÷àåì çíà÷åíèÿ ïîòîêîâûé ïåðåìåííûõ

					float cGP1 = edgeWGP[iGP] * baseF(iF, c1, edgeGP[i][iGP].x, edgeGP[i][iGP].y);
					float cGP2 = edgeWGP[iGP] * baseF(iF, c2, edgeGP[i][iGP].x, edgeGP[i][iGP].y);

					// èíòåãðàëà ïî ïîòîêó FU äîëæíî áûòü äâà - nx, ny
					tmpIntU1 += (edgeNormal[i].x * FWx * cGP1) + (edgeNormal[i].y * FWy * cGP1);

					tmpIntU2 += (edgeNormal[i].x * FWx * cGP2) + (edgeNormal[i].y * FWy * cGP2);
				}

				if (abs(tmpIntU1) <= EPS) tmpIntU1 = 0.0;

				//intU[c1][iF] += tmpIntU1 * edgeJ[i];
				intU[c1 * funcCount + iF] += tmpIntU1 * edgeJ[i];

				if (abs(tmpIntU2) <= EPS) tmpIntU2 = 0.0;

				//intU[c2][iF] -= tmpIntU2 * edgeJ[i];
				intU[c2 * funcCount + iF] -= tmpIntU2 * edgeJ[i];
			}
		}
		else { // C2 < 0, ãðàíè÷íàÿ ñòîðîíà
			isBoundary = true;

			for (int iF = 0; iF < funcCount; iF++) // êàæäûé ñëó÷àé ïðè ñêàëÿðíîì óìíîæåíèè óðàâíåíèé íà baseF( iF, ... )
			{
				float tmpIntU1 = 0.0; // äëÿ ÿ÷åéêè Ñ1
				for (int iGP = 0; iGP < edgeGPCount; iGP++)
				{
					getFields(Ul, Wxl, Wyl, c1, edgeGP[i][iGP].x, edgeGP[i][iGP].y, NEW);

					// Çäåñü áûëî ìíîãî êîäà...
					float tmp = 0;
					getFlux(FU, FWx, FWy, Ul, Wxl, Wyl, tmp, tmp, tmp, edgeNormal[i], isBoundary);

					float cGP1 = edgeWGP[iGP] * baseF(iF, c1, edgeGP[i][iGP].x, edgeGP[i][iGP].y);

					tmpIntU1 += (edgeNormal[i].x * FWx * cGP1) + (edgeNormal[i].y * FWy * cGP1);
				}

				if (abs(tmpIntU1) <= EPS) tmpIntU1 = 0.0;

				//intU[c1][iF] += tmpIntU1 * edgeJ[i];
				intU[c1 * funcCount + iF] += tmpIntU1 * edgeJ[i];
			}
		}
	}
}

// **************************************************************************************************************************************************

// ************************************************* Ðàáîòà ñ ìàòðèöîé âåñîâ ************************************************************************
void getMatrixInCell(int icell, float *A)
{
	int matrixDim = cellsCount * funcCount;

	int matrixRowStep = icell * funcCount * matrixDim;
	int matrixColStep = icell * funcCount;

	int matrAIndex = 0;

	for (int i = 0; i < funcCount; i++)
	{
		for (int j = 0; j < funcCount; j++)
		{
			matrAIndex = matrixRowStep + i * matrixDim + matrixColStep + j;
			A[i * funcCount + j] = matrA[matrAIndex];
		}

		//matrAIndex = matrixRowStep + (i + 1) * matrixDim + matrixColStep;
	}
}

void getInvMatrixInCell(int icell, float *invA)
{
	int matrixDim = cellsCount * funcCount;

	int matrixRowStep = icell * funcCount * matrixDim;
	int matrixColStep = icell * funcCount;

	int matrAIndex = 0;

	for (int i = 0; i < funcCount; i++)
	{
		for (int j = 0; j < funcCount; j++)
		{
			matrAIndex = matrixRowStep + i * matrixDim + matrixColStep + j;
			invA[i * funcCount + j] = matrInvA[matrAIndex];
		}

		//matrAIndex = matrixRowStep + (i + 1) * matrixDim + matrixColStep;
	}
}

void setMatrixInCell(int icell, float *A)
{
	int matrixDim = cellsCount * funcCount;

	int matrixRowStep = icell * funcCount * matrixDim;
	int matrixColStep = icell * funcCount;

	int matrAIndex = 0;

	for (int i = 0; i < funcCount; i++)
	{
		for (int j = 0; j < funcCount; j++)
		{
			matrAIndex = matrixRowStep + i * matrixDim + matrixColStep + j;
			matrA[matrAIndex] = A[i * funcCount + j];
		}

		//matrAIndex = matrixRowStep + (i + 1) * matrixDim + matrixColStep;
	}
}

void setInvMatrixInCell(int icell, float *invA)
{
	int matrixDim = cellsCount * funcCount;

	int matrixRowStep = icell * funcCount * matrixDim;
	int matrixColStep = icell * funcCount;

	int matrAIndex = 0;

	for (int i = 0; i < funcCount; i++)
	{
		for (int j = 0; j < funcCount; j++)
		{
			matrAIndex = matrixRowStep + i * matrixDim + matrixColStep + j;
			matrInvA[matrAIndex] = invA[i * funcCount + j];
		}

		//matrAIndex = matrixRowStep + (i + 1) * matrixDim + matrixColStep;
	}
}

void calcMatr(int iCell)
{
	//float *	A = matrA[iCell];
	//float *	invA = matrInvA[iCell];
	//тут это не нужно, надо просто брать "чистую" матрицу и передавать на основную и в inv
	//getMatrixInCell(iCell, cellA);
	//getInvMatrixInCell(iCell, cellInvA);

	int& N = funcCount;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			cellA[i * N + j] = 0.0;
			for (int iGP = 0; iGP < cellGPCount; iGP++)
			{
				//A[i][j] += cellWGP[iGP] * baseF(i, iCell, cellGP[iCell][iGP].x, cellGP[iCell][iGP].y)*baseF(j, iCell, cellGP[iCell][iGP].x, cellGP[iCell][iGP].y);
				//A[i][j] += cellWGP[iGP] * baseF(i, iCell, cellGP[iCell * cellGPCount + iGP].x, cellGP[iCell * cellGPCount + iGP].y)*baseF(j, iCell, cellGP[iCell * cellGPCount + iGP].x, cellGP[iCell * cellGPCount + iGP].y);
				cellA[i * N + j] += cellWGP[iGP] * baseF(i, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP])*baseF(j, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);
			}
			cellA[i * N + j] *= cellJ[iCell];
		}
	}

	setMatrixInCell(iCell, cellA);

	inverseMatr(cellA, cellInvA, N);

	setInvMatrixInCell(iCell, cellInvA);
}

void inverseMatr__(float** a_src, float **am, int N)
{
	float **a = a_src;
	float detA = a[0][0] * a[1][1] * a[2][2] + a[1][0] * a[2][1] * a[0][2] + a[0][1] * a[1][2] * a[2][0]
		- a[2][0] * a[1][1] * a[0][2] - a[1][0] * a[0][1] * a[2][2] - a[0][0] * a[2][1] * a[1][2];
	//OutPut("detA = %25.16e\n", detA);
	float m[3][3];
	m[0][0] = a[1][1] * a[2][2] - a[2][1] * a[1][2];
	m[0][1] = a[2][0] * a[1][2] - a[1][0] * a[2][2];
	m[0][2] = a[1][0] * a[2][1] - a[2][0] * a[1][1];
	m[1][0] = a[2][1] * a[0][2] - a[0][1] * a[2][2];
	m[1][1] = a[0][0] * a[2][2] - a[2][0] * a[0][2];
	m[1][2] = a[2][0] * a[0][1] - a[0][0] * a[2][1];
	m[2][0] = a[0][1] * a[1][2] - a[1][1] * a[0][2];
	m[2][1] = a[1][0] * a[0][2] - a[0][0] * a[1][2];
	m[2][2] = a[0][0] * a[1][1] - a[1][0] * a[0][1];

	am[0][0] = m[0][0] / detA;
	am[0][1] = m[1][0] / detA;
	am[0][2] = m[2][0] / detA;
	am[1][0] = m[0][1] / detA;
	am[1][1] = m[1][1] / detA;
	am[1][2] = m[2][1] / detA;
	am[2][0] = m[0][2] / detA;
	am[2][1] = m[1][2] / detA;
	am[2][2] = m[2][2] / detA;
}

void inverseMatr(float* a_src, float *am, int N)
{
	int	*	mask;
	float	fmaxval;
	int		maxind;
	int		tmpi;
	float	tmp;
	//float	a[N][N];

	float	**a;

	mask = new int[N];
	a = new float*[N];
	for (int i = 0; i < N; i++)
	{
		a[i] = new float[N];
		for (int j = 0; j < N; j++)
		{
			a[i][j] = a_src[i * N + j];
		}
	}
	//::memcpy(a, a_src, sizeof(float)*N*N);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (i == j)
			{
				am[i * N + j] = 1.0;
			}
			else {
				am[i * N + j] = 0.0;
			}
		}
	}
	for (int i = 0; i < N; i++)
	{
		mask[i] = i;
	}
	for (int i = 0; i < N; i++)
	{
		maxind = i;
		fmaxval = fabs(a[i][i]);
		for (int ni = i + 1; ni < N; ni++)
		{
			if (fabs(fmaxval) <= fabs(a[ni][i]))
			{
				fmaxval = fabs(a[ni][i]);
				maxind = ni;
			}
		}
		fmaxval = a[maxind][i];
		if (fmaxval == 0)
		{
			OutPut("ERROR! Determinant of mass matrix is zero...\n");
			return;
		}
		if (i != maxind)
		{
			for (int nj = 0; nj < N; nj++)
			{
				tmp = a[i][nj];
				a[i][nj] = a[maxind][nj];
				a[maxind][nj] = tmp;

				tmp = am[i * N + nj];
				am[i * N + nj] = am[maxind * N + nj];
				am[maxind * N + nj] = tmp;
			}
			tmpi = mask[i];
			mask[i] = mask[maxind];
			mask[maxind] = tmpi;
		}
		float aii = a[i][i];
		for (int j = 0; j < N; j++)
		{
			a[i][j] = a[i][j] / aii;
			am[i * N + j] = am[i * N + j] / aii;
		}
		for (int ni = 0; ni < N; ni++)
		{
			if (ni != i)
			{
				float fconst = a[ni][i];
				for (int nj = 0; nj < N; nj++)
				{
					a[ni][nj] = a[ni][nj] - fconst *  a[i][nj];
					am[ni * N + nj] = am[ni * N + nj] - fconst * am[i * N + nj];
				}
			}
		}
	}

	//for (int i = 0; i < N; i++)
	//{
	//	if (mask[i] != i) 
	//	{
	//		for (int j = 0; j < N; j++) 
	//		{
	//			tmp				= a[i][j];
	//			a[i][j]			= a[mask[i]][j];
	//			a[mask[i]][j]	= tmp;
	//		}
	//	}
	//}
	for (int i = 0; i < N; i++)
	{
		delete[] a[i];
	}
	delete[] a;
	delete[] mask;
	return;
}

// **************************************************************************************************************************************************

// ******************************************* Ïîëó÷åíèå âñïîìîãàòåëüíûõ çíà÷åíèé *******************************************************************
void getFlux(float& FU, float& FWx, float& FWy,
	float Ul, float Wxl, float Wyl,
	float Ur, float Wxr, float Wyr, Vector n, bool isBoundary) // Ïîëó÷àåì ïîòîêîâûå çíà÷åíèÿ.
{
	float c11 = 1.0;

	if (!isBoundary)
	{
		FU = (Ul + Ur) * 0.5;
		FWx = (Wxl + Wxr) * 0.5 + c11 * (Ur - Ul) * n.x;
		FWy = (Wyl + Wyr) * 0.5 + c11 * (Ur - Ul) * n.y;
	}
	else
	{
		FU = 0.0;
		FWx = Wxl + c11 * (FU - Ul) * n.x;
		FWy = Wyl + c11 * (FU - Ul) * n.y;
	}
}

void getFields(float& U, float& Wx, float& Wy, int iCell, float x, float y, DATA_LAYER layer) // // Ïî èíäåêñó êàæäîé ÿ÷åéêè ïîëó÷àåì çíà÷åíèå U, Wx, Wy
{
	float *fieldU, *fieldWx, *fieldWy;
	fieldU = (float*)malloc(funcCount * sizeof(float));
	fieldWx = (float*)malloc(funcCount * sizeof(float));
	fieldWy = (float*)malloc(funcCount * sizeof(float));

	switch (layer)
	{
	case OLD:
		for (int i = 0; i < funcCount; i++)
		{
			fieldU[i] = fUOld[iCell * funcCount + i];
			fieldWx[i] = fWxOld[iCell * funcCount + i];
			fieldWy[i] = fWyOld[iCell * funcCount + i];
		}
		break;
	case NEW:
	default:
		for (int i = 0; i < funcCount; i++)
		{
			fieldU[i] = fU[iCell * funcCount + i];
			fieldWx[i] = fWx[iCell * funcCount + i];
			fieldWy[i] = fWy[iCell * funcCount + i];
		}
		break;
	}

	U = fieldU[0];
	Wx = fieldWx[0];
	Wy = fieldWy[0];

	for (int j = 1; j < funcCount; j++)
	{
		float bF = baseF(j, iCell, x, y);
		//printf("%f\n", fieldU[j]);

		U += fieldU[j] * bF;
		Wx += fieldWx[j] * bF;
		Wy += fieldWy[j] * bF;
	}

	free(fieldU);
	free(fieldWx);
	free(fieldWy);

	//óäàëèòü fieldu...
}

// **************************************************************************************************************************************************

// ******************************************* Íåïîñðåäñòâåííî âû÷èñëåíèå çíà÷åíèé ******************************************************************
void calcNewGradientValues()
{
	float *aWx, *aWy;
	aWx = (float*)malloc(sizeof(float) * cellsCount * funcCount);
	aWy = (float*)malloc(sizeof(float) * cellsCount * funcCount);

	/*cublasGradMatrixMult(aWx, aWy);

	for (int iCell = 0; iCell < cellsCount; iCell++)
	{
		for (int j = 0; j < funcCount; j++)
		{
			fWx[iCell * funcCount + j] += TAU * aWx[iCell * funcCount + j];
			fWy[iCell * funcCount + j] += TAU * aWy[iCell * funcCount + j];

			if (abs(fWx[iCell * funcCount + j]) <= EPS) fWx[iCell * funcCount + j] = 0.0;
			if (abs(fWy[iCell * funcCount + j]) <= EPS) fWy[iCell * funcCount + j] = 0.0;
		}
	}

	free(aWx);
	free(aWy);*/

	for (int iCell = 0; iCell < cellsCount; iCell++)
	{
		//float * A = matrInvA[iCell];
		getInvMatrixInCell(iCell, cellInvA);

		memset(aWx, 0, sizeof(float)*funcCount);
		memset(aWy, 0, sizeof(float)*funcCount);

		for (int j = 0; j < funcCount; j++)
		{
			for (int k = 0; k < funcCount; k++)
			{
				//aWx[j] += A[j][k] * intWx[iCell][k];
				aWx[j] += cellInvA[j * funcCount + k] * intWx[iCell * funcCount + k];
				//aWy[j] += A[j][k] * intWy[iCell][k];
				aWy[j] += cellInvA[j * funcCount + k] * intWy[iCell * funcCount + k];
			}
		}

		for (int j = 0; j < funcCount; j++)
		{
			fWx[iCell * funcCount + j] += TAU * aWx[j];
			fWy[iCell * funcCount + j] += TAU * aWy[j];

			if (abs(fWx[iCell * funcCount + j]) <= EPS) fWx[iCell * funcCount + j] = 0.0;
			if (abs(fWy[iCell * funcCount + j]) <= EPS) fWy[iCell * funcCount + j] = 0.0;
		}
	}

	free(aWx);
	free(aWy);
}

void calcNewValues()
{
	float *aU;
	aU = (float*)malloc(sizeof(float) * funcCount);

	for (int iCell = 0; iCell < cellsCount; iCell++)
	{
		//float * A = matrInvA[iCell];

		getInvMatrixInCell(iCell, cellInvA);

		memset(aU, 0, sizeof(float) * funcCount);

		for (int j = 0; j < funcCount; j++)
		{
			for (int k = 0; k < funcCount; k++)
			{
				//aU[j] += A[j][k] * intU[iCell][k];
				aU[j] += cellInvA[j * funcCount + k] * intU[iCell * funcCount + k];
			}
		}

		for (int j = 0; j < funcCount; j++)
		{
			fU[iCell * funcCount + j] += TAU * aU[j];

			if (abs(fU[iCell * funcCount + j]) <= EPS)
			{
				fU[iCell * funcCount + j] = 0.0;
			}
		}
	}

	/*float *aU;
	aU = (float*)malloc(sizeof(float) * cellsCount * funcCount);

	cublasMatrixMult(aU);

	for (int iCell = 0; iCell < cellsCount; iCell++)
	{
		for (int j = 0; j < funcCount; j++)
		{
			fU[iCell * funcCount + j] += TAU * aU[iCell * funcCount + j];

			if (abs(fU[iCell * funcCount + j]) <= EPS)
			{
				fU[iCell * funcCount + j] = 0.0;
			}
		}
	}*/

	free(aU);
}

// **************************************************************************************************************************************************

// ************************************************ Âñïîìîãàòåëüíûå ôóíêöèè *************************************************************************

void copyToOld()
{
	for (int i = 0; i < cellsCount; i++)
	{
		/*memcpy(fUOld[i], fU[i], funcCount * sizeof(float));
		memcpy(fWxOld[i], fWx[i], funcCount * sizeof(float));
		memcpy(fWy[i], fWy[i], funcCount * sizeof(float));*/

		for (int j = 0; j < funcCount; j++)
		{
			fUOld[i * funcCount + j] = fU[i * funcCount + j];
			fWxOld[i * funcCount + j] = fWx[i * funcCount + j];
			fWyOld[i * funcCount + j] = fWy[i * funcCount + j];
		}
	}

	memcpy(uOld, u, cellsCount * sizeof(float));
	memcpy(wxOld, wx, cellsCount * sizeof(float));
	memcpy(wyOld, wy, cellsCount * sizeof(float));
}

float max(float a, float b)
{
	if (a > b) return a;
	return b;
}

float max(float a, float b, float c)
{
	return max(max(a, b), c);
}

float min(float a, float b, float c)
{
	return min(min(a, b), c);
}

float min(float a, float b)
{
	if (a < b) return a;
	return b;
}

void calcResidual()
{
	float absU, absWx, absWy;
	resU = 0.0;
	resWx = 0.0;
	resWy = 0.0;

	for (int i = 0; i < cellsCount; i++)
	{
		absU = abs(u[i] - uOld[i]);

		float utmp, wxtmp, wytmp;
		getFields(utmp, wxtmp, wytmp, i, cellCx[i], cellCy[i], NEW);
		//printf("U = %f\n", utmp);
		//int c;
		//scanf("%d", &c);

		absWx = abs(wx[i] - wxOld[i]);
		absWy = abs(wy[i] - wyOld[i]);

		if (resU < absU) resU = absU;
		if (resWx < absWx) resWx = absWx;
		if (resWy < absWy) resWy = absWy;
	}
}

// **************************************************************************************************************************************************

// *************************************************** Îñâîáîæäàåì ïàìÿòü ***************************************************************************
void destroyData()
{
	free(u);
	free(wx);
	free(wy);
	free(uOld);
	free(wxOld);
	free(wyOld);



	for (int i = 0; i < cellsCount; i++)
	{
		//free(cellGP[i]);
		//free(edgeGP[i]);
		/*free(fU[i]);
		free(fWx[i]);
		free(fWy[i]);
		free(fUOld[i]);
		free(fWxOld[i]);
		free(fWyOld[i]);*/
		//free(intU[i]);
		//free(intWx[i]);
		//free(intWy[i]);

		/*for (int j = 0; j < funcCount; j++)
		{
		free(matrA[i][j]);
		free(matrInvA[i][j]);
		}*/

		//free(matrA[i]);
		//free(matrInvA[i]);
	}

	for (int i = 0; i < edgesCount; i++)
	{
		free(edgeGP[i]);
	}


	free(matrA);
	free(matrInvA);

	free(cellA);
	free(cellInvA);

	free(intU);
	free(intWx);
	free(intWy);
	free(edgeGP);
	free(cellGPx);
	free(cellGPy);
	free(cellWGP);
	free(edgeWGP);
	free(fU);
	free(fWx);
	free(fWy);
	free(fWxOld);
	free(fWyOld);
	free(fUOld);

	free(cellJ);
	free(edgeJ);
}

void destroyGrid()
{
	for (int i = 0; i < cellsCount; i++)
	{
		free(cellNodes[i]);
		free(cellNeigh[i]);
		free(cellEdges[i]);
	}
	free(cellNodes);
	free(cellEdges);
	free(cellS);
	free(cellIsBound);
	free(cellCx);
	free(cellCy);

	//cudaFree(cellCx_dev);
	//cudaFree(cellCy_dev);
	//cudaFree(cellDXx_dev);
	//cudaFree(cellDXy_dev);

	free(cellType);
	free(edgeType);
	free(edgeNode1);
	free(edgeNode2);
	free(edgeCell1);
	free(edgeCell2);
	free(edgeNormal);
	free(edgeC);
	free(edgeL);
}

// **************************************************************************************************************************************************

// ***************************************************** Âûâîä â ôàéëû ******************************************************************************
void writeVTK_AMGX(char* name, int step, float* currentLayerSolution)
{
    char fName[50];
    FILE * fp;
    sprintf(fName, "%s.%010d.vtk", name, step);
    fp = fopen(fName, "w");
    fprintf(fp, "");
    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "2D RKDG method for task '%s' results.\n", name);
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");

    fprintf(fp, "POINTS %d float\n", nodesCount);
    for (int i = 0; i < nodesCount; i++)
    {
        fprintf(fp, "%f %f %f  \n", nodes[i].x, nodes[i].y, 0.0);
    }

    fprintf(fp, "\n");

    fprintf(fp, "CELLS %d %d\n", cellsCount, 4 * cellsCount);
    for (int i = 0; i < cellsCount; i++)
    {
        fprintf(fp, "3 %d %d %d  \n", cellNodes[i][0], cellNodes[i][1], cellNodes[i][2]);
    }

    fprintf(fp, "\n");

    fprintf(fp, "CELL_TYPES %d\n", cellsCount);

    for (int i = 0; i < cellsCount; i++) fprintf(fp, "5\n");

    fprintf(fp, "\n");

    fprintf(fp, "CELL_DATA %d\n", cellsCount);

    fprintf(fp, "SCALARS Temperature float 1\nLOOKUP_TABLE default\n");
    for (int i = 0; i < cellsCount; i++)
    {
        fprintf(fp, "%f ", currentLayerSolution[i * A_block_size + 6]);
        if ((i + 1) % 8 == 0 || i + 1 == cellsCount) fprintf(fp, "\n");
    }

    ////fprintf(fp, "SCALARS Pressure float 1\nLOOKUP_TABLE default\n");
    //for (int i = 0; i < cellsCount; i++)
    //{
    //	fprintf(fp, "%f ", wx[i]);
    //	if ((i + 1) % 8 == 0 || i + 1 == cellsCount) fprintf(fp, "\n");
    //}

    /*fprintf(fp, "SCALARS Energy float 1\nLOOKUP_TABLE default\n");
    for (int i = 0; i < cellsCount; i++)
    {
    fprintf(fp, "%f ", wy[i]);
    if ((i + 1) % 8 == 0 || i + 1 == cellsCount) fprintf(fp, "\n");
    }*/

    /*fprintf(fp, "SCALARS Mach_number float 1\nLOOKUP_TABLE default\n");
    for (int i = 0; i < cellsCount; i++)
    {
    fprintf(fp, "%f ", sqrt((u[i] * u[i] + v[i] * v[i]) / (GAM*p[i] / r[i])));
    if ((i + 1) % 8 == 0 || i + 1 == cellsCount) fprintf(fp, "\n");
    }*/

    /*fprintf(fp, "VECTORS Velosity float\n");
    for (int i = 0; i < cellsCount; i++)
    {
    fprintf(fp, "%f %f %f  ", u[i], v[i], 0.0);
    if ((i + 1) % 8 == 0 || i + 1 == cellsCount) fprintf(fp, "\n");
    }*/

    fclose(fp);
}

void writeVTK(char* name, int step)
{
	char fName[50];
	FILE * fp;
	sprintf(fName, "%s.%010d.vtk", name, step);
	fp = fopen(fName, "w");
	fprintf(fp, "");
	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "2D RKDG method for task '%s' results.\n", name);
	fprintf(fp, "ASCII\n");
	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");

	fprintf(fp, "POINTS %d float\n", nodesCount);
	for (int i = 0; i < nodesCount; i++)
	{
		fprintf(fp, "%f %f %f  \n", nodes[i].x, nodes[i].y, 0.0);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELLS %d %d\n", cellsCount, 4 * cellsCount);
	for (int i = 0; i < cellsCount; i++)
	{
		fprintf(fp, "3 %d %d %d  \n", cellNodes[i][0], cellNodes[i][1], cellNodes[i][2]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", cellsCount);
	for (int i = 0; i < cellsCount; i++) fprintf(fp, "5\n");
	fprintf(fp, "\n");

	fprintf(fp, "CELL_DATA %d\n", cellsCount);

	fprintf(fp, "SCALARS Temperature float 1\nLOOKUP_TABLE default\n");
	for (int i = 0; i < cellsCount; i++)
	{
		fprintf(fp, "%f ", fU[i * funcCount + 0]);
		if ((i + 1) % 8 == 0 || i + 1 == cellsCount) fprintf(fp, "\n");
	}

	////fprintf(fp, "SCALARS Pressure float 1\nLOOKUP_TABLE default\n");
	//for (int i = 0; i < cellsCount; i++)
	//{
	//	fprintf(fp, "%f ", wx[i]);
	//	if ((i + 1) % 8 == 0 || i + 1 == cellsCount) fprintf(fp, "\n");
	//}

	/*fprintf(fp, "SCALARS Energy float 1\nLOOKUP_TABLE default\n");
	for (int i = 0; i < cellsCount; i++)
	{
	fprintf(fp, "%f ", wy[i]);
	if ((i + 1) % 8 == 0 || i + 1 == cellsCount) fprintf(fp, "\n");
	}*/

	/*fprintf(fp, "SCALARS Mach_number float 1\nLOOKUP_TABLE default\n");
	for (int i = 0; i < cellsCount; i++)
	{
	fprintf(fp, "%f ", sqrt((u[i] * u[i] + v[i] * v[i]) / (GAM*p[i] / r[i])));
	if ((i + 1) % 8 == 0 || i + 1 == cellsCount) fprintf(fp, "\n");
	}*/

	/*fprintf(fp, "VECTORS Velosity float\n");
	for (int i = 0; i < cellsCount; i++)
	{
	fprintf(fp, "%f %f %f  ", u[i], v[i], 0.0);
	if ((i + 1) % 8 == 0 || i + 1 == cellsCount) fprintf(fp, "\n");
	}*/

	fclose(fp);
}

void save(int step)
{
	char fName[50];
	FILE * fp;

	sprintf(fName, "res_%010d.gnuplot.txt", step);
	fp = fopen(fName, "w");
	for (int i = 0; i < 300; i++)
	{
		for (int j = 0; j < 100; j++)
		{
			float	x = 3.0*i / 300;
			float	y = 1.0*j / 100;
			if (x > 0.6 && y < 0.2)
			{
				fprintf(fp, "%16.8e %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e\n", x, y, 0.0, 0.0, 0.0, 0.0, 0.0);
			}
			else
			{
				float	minL = 1.e+10;
				int		minI = 0;
				for (int iC = 0; iC < cellsCount; iC++)
				{
					float &ptx = cellCx[iC];
					float &pty = cellCy[iC];

					float L = sqrt(pow_2(x - ptx) + pow_2(y - pty));
					if (L < minL)
					{
						minL = L;
						minI = iC;
					}
				}
				float uCurrent, wxCurrent, wyCurrent;
				getFields(uCurrent, wxCurrent, wyCurrent, minI, x, y, NEW);

				fprintf(fp, "%16.8e %16.8e %16.8e %16.8e %16.8e\n", x, y, uCurrent, wxCurrent, wyCurrent);
			}
		}
	}
	fclose(fp);


	sprintf(fName, "res_%010d.cells.txt", step);
	fp = fopen(fName, "w");
	for (int j = 0; j < cellsCount; j++)
	{
		float uCurrent, wxCurrent, wyCurrent;
		getFields(uCurrent, wxCurrent, wyCurrent, j, cellCx[j], cellCy[j], NEW);

		fprintf(fp, "%d %16.8e %16.8e %16.8e %16.8e %16.8e\n", j, cellCx[j], cellCy[j], uCurrent, wxCurrent, wyCurrent);
	}
	fclose(fp);
}

void OutPut(const char* str, ...)
{
	va_list list;

	va_start(list, str);
	vprintf(str, list);
	::fflush(stdout);
	va_end(list);

	if (fileLog) {
		va_start(list, str);
		vfprintf(fileLog, str, list);
		va_end(list);
		::fflush(fileLog);
	} // if
} // OutPut



float GetUBoundary(int iCell, int iCoeff)
{
	switch (iCoeff)
	{
		case 0:
			return 0.0;
        case 1:
            return 0.0;
	    case 2:
	        return 0.0;
		default:
			return 0.0;
	}
}

float GetQxBoundary(int iCell, int iCoeff)
{
    switch (iCoeff)
    {
        case 0:
            return 0.0;
        case 1:
            return 0.0;
        case 2:
            return 0.0;
        default:
            return 0.0;
    }
}

float GetQyBoundary(int iCell, int iCoeff)
{
    switch (iCoeff)
    {
        case 0:
            return 0.0;
        case 1:
            return 0.0;
        case 2:
            return 0.0;
        default:
            return 0.0;
    }
}

void initRightPart(float *lastLayerSolution, float *rightPart_data)
{
    const int qxBlock = 0;
    const int qyBlock = 1;
    const int uBlock = 2;

    int iCell = 0;
    int iSmallBlockId = 0;

    A_size = A_block_size * cellsCount;

    for (int i = 0; i < A_size; i++) {

        if (((i % A_small_block_size) == 0) && (i != 0)) {
            iSmallBlockId++;
        }

        if (((i % A_block_size) == 0) && (i != 0)) {
            iCell++;
            iSmallBlockId = 0;
        }

    	// rowId in each small block
        int rowId = i % A_small_block_size;

        float fullInt = 0.0;

        if (iSmallBlockId == uBlock) {
            for (int j = 0; j < A_small_block_size; j++)
            {
                float tmpInt = 0.0;

                int iRowU = (iCell * A_block_size + 6) + j;

                for (int iGP = 0; iGP < cellGPCount; iGP++)
                {
                    tmpInt += cellWGP[iGP] * baseF(rowId, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]) * baseF(j, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);
                }

                tmpInt *= (cellJ[iCell] * lastLayerSolution[iRowU]);

                tmpInt /= TAU;

                fullInt += tmpInt;
            }

            rightPart_data[i] = fullInt;
        }
    }
}

// initialization matrix for implicit method
void initMatrix(CSRMatrix& A)
{
	const int qxBlock = 0;
	const int qyBlock = 1;
	const int uBlock = 2;

	A_size = A_block_size * cellsCount;	// 9 * cellsCount

	//CSRMatrix A(A_size);

	int iCell = 0;
	int iSmallBlockId = 0;

	// begin matrix initialization
	for (int i = 0; i < A_size; i++)
	{
        if ((i % A_small_block_size) == 0 && (i != 0))
        {
            iSmallBlockId++;
        }

        if ((i % A_block_size) == 0 && (i != 0))
        {
            iCell++;
            iSmallBlockId = 0;
        }

	    // variable for current colum
		int colId = iCell * A_block_size + iSmallBlockId * A_small_block_size;

	    // rowId in each small block
		int rowId = i % A_small_block_size;	

		// сalculate the same values of the matrix A
		for (int j = 0; j < A_small_block_size; j++)
		{
			float tmpInt = 0.0;
			A.set(i, colId + j, 0.0);

			for (int iGP = 0; iGP < cellGPCount; iGP++)
			{
				tmpInt += cellWGP[iGP] * baseF(rowId, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]) * baseF(j, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);
			}

			tmpInt *= cellJ[iCell];

			if (iSmallBlockId == uBlock)
			{
				tmpInt /= TAU;
			}

			A.set(i, colId + j, tmpInt);
		}

		// calculate vol integral values for qx-block
		/*if (iSmallBlockId == qxBlock)
		{
			for (int j = 0; j < A_small_block_size; j++)
			{
				float tmpInt = 0.0;

				for (int iGP = 0; iGP < cellGPCount; iGP++)
				{
					float bfDFDx = baseDFDx(rowId, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);

					tmpInt += cellWGP[iGP] * bfDFDx * baseF(j, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);
				}

				if (abs(tmpInt) <= EPS) tmpInt = 0.0;

				tmpInt = tmpInt * cellJ[iCell];

				A.add(i, colId + j, tmpInt);
			}
		}
		// calculate vol integral values for qy-block
		else if (iSmallBlockId == qyBlock) 
		{
			for (int j = 0; j < A_small_block_size; j++)
			{
				float tmpInt = 0.0;

				for (int iGP = 0; iGP < cellGPCount; iGP++)
				{
					float bfDFDy = baseDFDy(rowId, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);

					tmpInt += cellWGP[iGP] * bfDFDy * baseF(j, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);
				}

				if (abs(tmpInt) <= EPS) tmpInt = 0.0;

				tmpInt = tmpInt * cellJ[iCell];

				A.add(i, colId + j, tmpInt);
			}
		}
		// calculate vol integral values for u-block
		else if (iSmallBlockId == uBlock)
		{
			for (int j = 0; j < A_small_block_size; j++)
			{
				float tmpInt = 0.0;

				for (int iGP = 0; iGP < cellGPCount; iGP++)
				{
					float bfDFDx = baseDFDx(rowId, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);
					float bfDFDy = baseDFDy(rowId, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);

					tmpInt += cellWGP[iGP] * (bfDFDx + bfDFDy) * baseF(j, iCell, cellGPx[iCell * cellGPCount + iGP], cellGPy[iCell * cellGPCount + iGP]);
				}

				if (abs(tmpInt) <= EPS) tmpInt = 0.0;

				tmpInt = tmpInt * cellJ[iCell];

				A.add(i, colId + j, tmpInt);
			}
		}*/
	}

	// calculate surf integral
	/*int c1, c2;
	bool isBoundary = false;
	for (int i = 0; i < edgesCount; i++)
	{
		c1 = edgeCell1[i];	
		c2 = edgeCell2[i];	
		if (c2 >= 0) {
			isBoundary = false;
			int iRowU1 = c1 * A_block_size + 6;
			int iRowU2 = c2 * A_block_size + 6;
			int iRowQy1 = c1 * A_block_size + 3;
			int iRowQy2 = c2 * A_block_size + 3;
			int iRowQx1 = c1 * A_block_size;
			int iRowQx2 = c2 * A_block_size;

			int iColU1 = c1 * A_block_size + uBlock * A_small_block_size;
            int iColU2 = c2 * A_block_size + uBlock * A_small_block_size;
            int iColQy1 = c1 * A_block_size + qyBlock * A_small_block_size;
            int iColQy2 = c2 * A_block_size + qyBlock * A_small_block_size;
            int iColQx1 = c1 * A_block_size + qxBlock * A_small_block_size;
            int iColQx2 = c2 * A_block_size + qxBlock * A_small_block_size;

			for (int m = 0; m < funcCount; m++)
			{
				for (int j = 0; j < A_small_block_size; j++)
				{
					int iRowU1Current = iRowU1 + m;
					int iRowU2Current = iRowU2 + m;
					int iRowQx1Current = iRowQx1 + m;
					int iRowQx2Current = iRowQx2 + m;
					int iRowQy1Current = iRowQy1 + m;
					int iRowQy2Current = iRowQy2 + m;

					float tmpIntU1 = 0.0, tmpIntQx1 = 0.0, tmpIntQy1 = 0.0;
					float tmpIntU2 = 0.0, tmpIntQx2 = 0.0, tmpIntQy2 = 0.0;

					for (int iGP = 0; iGP < edgeGPCount; iGP++)
					{
						float cGP1 = edgeWGP[iGP] * baseF(m, c1, edgeGP[i][iGP].x, edgeGP[i][iGP].y) * baseF(j, c1, edgeGP[i][iGP].x, edgeGP[i][iGP].y);
						float cGP2 = edgeWGP[iGP] * baseF(m, c2, edgeGP[i][iGP].x, edgeGP[i][iGP].y) * baseF(j, c2, edgeGP[i][iGP].x, edgeGP[i][iGP].y);

						tmpIntU1 += (edgeNormal[i].x * cGP1) + (edgeNormal[i].y * cGP1);
						tmpIntU2 += (edgeNormal[i].x * cGP2) + (edgeNormal[i].y * cGP2);

						tmpIntQx1 += edgeNormal[i].x * cGP1;
						tmpIntQx2 += edgeNormal[i].x * cGP2;

						tmpIntQy1 += edgeNormal[i].y * cGP1;
						tmpIntQy2 += edgeNormal[i].y * cGP2;
					}

					if (abs(tmpIntU1) <= EPS) tmpIntU1 = 0.0;

					// for u
					tmpIntU1 *= edgeJ[i] * 0.5;
					A.add(iRowU1Current, iColU1 + j, -tmpIntU1);

					if (abs(tmpIntU2) <= EPS) tmpIntU2 = 0.0;

					tmpIntU2 *= edgeJ[i] * 0.5;
					A.add(iRowU2Current, iColU2 + j, -tmpIntU2);

					// for qx
					if (abs(tmpIntQx1) <= EPS) tmpIntQx1 = 0.0;

					tmpIntQx1 *= edgeJ[i] * 0.5;
					A.add(iRowQx1Current, iColQx1 + j, -tmpIntQx1);

					if (abs(tmpIntQx2) <= EPS) tmpIntQx2 = 0.0;

					tmpIntQx2 *= edgeJ[i] * 0.5;
					A.add(iRowQx2Current, iColQx2 + j, -tmpIntQx2);

					// for qy
					if (abs(tmpIntQy1) <= EPS) tmpIntQy1 = 0.0;

					tmpIntQy1 *= edgeJ[i] * 0.5;
					A.add(iRowQy1Current, iColQy1 + j, -tmpIntQy1);

					if (abs(tmpIntQy2) <= EPS) tmpIntQy2 = 0.0;

					tmpIntQy2 *= edgeJ[i] * 0.5;
					A.add(iRowQy2Current, iColQy2 + j, -tmpIntQy2);
				}
			}
		}
		else { 
			isBoundary = true;
			int iRowU1 = c1 * A_block_size + 6;
            int iColU1 = c1 * A_block_size + uBlock * A_small_block_size;

			for (int m = 0; m < funcCount; m++) 
			{
				for (int j = 0; j < A_small_block_size; j++)
				{
					int iRowU1Current = iRowU1 + m;

					float tmpIntU1 = 0.0;

					for (int iGP = 0; iGP < edgeGPCount; iGP++)
					{
						float cGP1 = edgeWGP[iGP] * baseF(m, c1, edgeGP[i][iGP].x, edgeGP[i][iGP].y) * baseF(j, c1, edgeGP[i][iGP].x, edgeGP[i][iGP].y);

						tmpIntU1 += (edgeNormal[i].x * cGP1) + (edgeNormal[i].y * cGP1);
					}

					if (abs(tmpIntU1) <= EPS) tmpIntU1 = 0.0;

					tmpIntU1 *= edgeJ[i] * 0.5;
					A.add(iRowU1Current, iColU1 + j, -tmpIntU1);
				}
			}
		}
	}*/
}