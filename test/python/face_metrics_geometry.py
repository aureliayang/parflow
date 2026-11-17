"""Standalone C-contract test for face PFB loading and shared flow geometry.

Run with Python 3 and clang/cc. Tests the actual C reader and geometry header
with minimal grid/vector shims, including partition halos and invalid input.
No ParFlow executable or MPI installation is needed.
"""
from pathlib import Path
import os
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SHIMS = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
typedef struct { int ix,iy,iz,nx,ny,nz; } Subgrid;
typedef struct { Subgrid *items[16]; int size; } SubgridArray;
typedef struct { SubgridArray *local,*all; int compute_ready; } Grid;
typedef struct { Subgrid box; double *data; } Subvector;
typedef struct { Grid *grid; Subvector *s[16]; } Vector;
typedef struct { int nx,ny,nz; } Background;
Background background={4,3,2};
#define GlobalsBackground (&background)
#define BackgroundNX(b) ((b)->nx)
#define BackgroundNY(b) ((b)->ny)
#define BackgroundNZ(b) ((b)->nz)
#define SubgridNX(s) ((s)->nx)
#define SubgridNY(s) ((s)->ny)
#define GridAllSubgrids(g) ((g)->all)
#define GridSubgrids(g) ((g)->local)
#define ForSubgridI(i,a) for(i=0;i<(a)->size;i++)
#define SubgridArraySubgrid(a,i) ((a)->items[i])
#define VectorSubvector(v,i) ((v)->s[i])
#define SubvectorIX(s) ((s)->box.ix)
#define SubvectorIY(s) ((s)->box.iy)
#define SubvectorIZ(s) ((s)->box.iz)
#define SubvectorNX(s) ((s)->box.nx)
#define SubvectorNY(s) ((s)->box.ny)
#define SubvectorNZ(s) ((s)->box.nz)
#define SubvectorData(s) ((s)->data)
#define SubvectorEltIndex(s,i,j,k) (((k)-(s)->box.iz)*(s)->box.ny*(s)->box.nx+((j)-(s)->box.iy)*(s)->box.nx+(i)-(s)->box.ix)
#define vector_cell_centered 0
#define pfmin(a,b) ((a)<(b)?(a):(b))
#define pfmax(a,b) ((a)>(b)?(a):(b))
SubgridArray *NewSubgridArray(void){return calloc(1,sizeof(SubgridArray));}
Subgrid *DuplicateSubgrid(Subgrid *s){Subgrid *r=malloc(sizeof(*r));*r=*s;return r;}
void AppendSubgrid(Subgrid *s,SubgridArray *a){a->items[a->size++]=s;}
SubgridArray *GetGridSubgrids(SubgridArray *a){return a;}
Grid *NewGrid(SubgridArray *l,SubgridArray *a){Grid *g=malloc(sizeof(*g));g->local=l;g->all=a;g->compute_ready=0;return g;}
void CreateComputePkgs(Grid *g){g->compute_ready=1;}
Vector *NewVectorType(Grid *g,int nc,int halo,int type){
 (void)nc;(void)type;assert(g->compute_ready);Vector *v=calloc(1,sizeof(*v));v->grid=g;
 for(int i=0;i<g->local->size;i++){
 Subvector *s=v->s[i]=calloc(1,sizeof(*s));s->box=*g->local->items[i];
 s->box.ix-=halo;s->box.iy-=halo;s->box.iz-=halo;
 s->box.nx+=2*halo;s->box.ny+=2*halo;s->box.nz+=2*halo;
 s->data=malloc(sizeof(double)*s->box.nx*s->box.ny*s->box.nz);}
 return v;}
void InitVectorAll(Vector *v,double val){for(int i=0;i<v->grid->local->size;i++){
 Subvector *s=v->s[i];for(int n=0;n<s->box.nx*s->box.ny*s->box.nz;n++)s->data[n]=val;}}
void CheckSphericalGeometry(int bad,const char *message){if(bad){fprintf(stderr,"%s\n",message);exit(2);}}
typedef struct {Vector *face_dx_scale,*face_dy_scale;double reference_radius;} ProblemData;
#define ProblemDataReferenceRadius(p) ((p)->reference_radius)
'''
MAIN = r'''
int main(int argc,char **argv){
 assert(argc==4);int ones=atoi(argv[3]);
 /* Repeat for two artificial ranks split in x and y, and two local blocks. */
 for(int split=0;split<3;split++){
 SubgridArray *a=NewSubgridArray();
 if(split==0){Subgrid s={0,0,0,2,3,2};AppendSubgrid(DuplicateSubgrid(&s),a);}
 if(split==1){Subgrid s={2,1,0,2,2,2};AppendSubgrid(DuplicateSubgrid(&s),a);}
 if(split==2){Subgrid s={0,0,0,2,3,2},t={2,0,0,2,3,2};AppendSubgrid(DuplicateSubgrid(&s),a);AppendSubgrid(DuplicateSubgrid(&t),a);}
 Grid *g=NewGrid(a,a);ProblemData pd={0};
 pd.face_dx_scale=ReadFaceMetric(g,argv[1],1);
 pd.face_dy_scale=ReadFaceMetric(g,argv[2],0);
 for(int is=0;is<a->size;is++){
 Subgrid *s=a->items[is];
 for(int k=s->iz;k<s->iz+s->nz;k++)for(int j=s->iy;j<s->iy+s->ny;j++)for(int i=s->ix;i<s->ix+s->nx;i++){
 PFHorizontalGeometry h=PFGetHorizontalGeometry(&pd,is,i,j,k,100,200,10,0,0,0,0);
 double xs=ones?1:1+0.1*j+0.01*k,xn=ones?1:1+0.1*(j+1)+0.01*k;
 double yw=ones?1:1+0.03*i,ye=ones?1:1+0.03*(i+1);
 assert(fabs(h.dx-50*(xs+xn))<1e-10);
 assert(fabs(h.dy-100*(yw+ye))<1e-10);
 assert(fabs(h.south-1000*xs)<1e-10);
 assert(fabs(h.north-1000*xn)<1e-10);
 assert(fabs(h.west-2000*yw)<1e-10);
 assert(fabs(h.east-2000*ye)<1e-10);
 assert(fabs(h.volume-h.dx*h.dy*10)<1e-8);
 /* Exterior and cross-partition halos are populated. */
 assert(isfinite(PFFaceScale(pd.face_dx_scale,is,i-1,j-1,k-1)));
 }
 }
 }
 /* Legacy reference-sphere equations remain unchanged. */
 ProblemData pd={0};pd.reference_radius=6371000;
 double dp=.001,dl=.002,ps=.6,pc=ps+.5*dp;
 PFHorizontalGeometry h=PFGetHorizontalGeometry(&pd,0,0,0,0,1,1,10,dl,ps,dp,0);
 assert(fabs(h.dx-6371000*cos(pc)*dl)<1e-9);
 assert(fabs(h.horizontal-6371000.*6371000*dl*2*cos(pc)*sin(.5*dp))<1e-5);
 puts("face PFB halos and geometry passed");return 0;
}
'''


def write_pfb(path, nx, ny, nz, value):
    # Multiple x subgrids exercise general whole-file PFB layout.
    with open(path, "wb") as f:
        f.write(struct.pack(">3d3i3di", 0, 0, 0, nx, ny, nz, 100, 200, 10, 2))
        for start, width in ((0, 2), (2, nx - 2)):
            f.write(struct.pack(">9i", start, 0, 0, width, ny, nz, 0, 0, 0))
            for k in range(nz):
                for j in range(ny):
                    for i in range(start, start + width):
                        f.write(struct.pack(">d", value(i, j, k)))


def main():
    source = (ROOT / "pfsimulator/parflow_lib/set_problem_data.c").read_text()
    reader = source[source.index("static unsigned long long FacePFBWord"):
                    source.index("static void PrepareFaceMetrics")]
    with tempfile.TemporaryDirectory() as directory:
        d = Path(directory)
        (d / "test.c").write_text(SHIMS + reader + '\n#include "face_metrics.h"\n' + MAIN)
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-fsanitize=address,undefined",
                        "-I" + str(ROOT / "pfsimulator/parflow_lib"),
                        str(d / "test.c"), "-lm", "-o", str(d / "test")], check=True)
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0")
        for ones in (True, False):
            write_pfb(d / "dx.pfb", 4, 4, 2, lambda i, j, k: 1 if ones else 1 + .1*j + .01*k)
            write_pfb(d / "dy.pfb", 5, 3, 2, lambda i, j, k: 1 if ones else 1 + .03*i)
            subprocess.run([str(d / "test"), str(d / "dx.pfb"), str(d / "dy.pfb"), str(int(ones))], check=True, env=env)
        for bad in (0, -1, float("nan")):
            write_pfb(d / "dx.pfb", 4, 4, 2, lambda *args: bad)
            result = subprocess.run([str(d / "test"), str(d / "dx.pfb"), str(d / "dy.pfb"), "0"], env=env, capture_output=True)
            assert result.returncode == 2, result.stderr
        write_pfb(d / "dx.pfb", 5, 4, 2, lambda *args: 1)
        result = subprocess.run([str(d / "test"), str(d / "dx.pfb"), str(d / "dy.pfb"), "0"], env=env, capture_output=True)
        assert result.returncode == 2, result.stderr
        print("invalid PFB values and dimensions rejected")


if __name__ == "__main__":
    main()
