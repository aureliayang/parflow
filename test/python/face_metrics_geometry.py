"""Standalone C-contract test for face PFB loading and cached flow geometry.

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
typedef struct { int nx,ny,nz; double dx,dy,dz,y; } Background;
Background background={4,3,2,100,200,20,0};
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
void CheckFaceMetrics(int bad,const char *message){if(bad){fprintf(stderr,"%s\n",message);exit(2);}}
typedef struct {Vector *face_dx_scale,*face_dy_scale,*zmult;Vector *face_geom_dx;Vector *face_geom_dy;Vector *face_geom_west;Vector *face_geom_east;Vector *face_geom_south;Vector *face_geom_north;Vector *face_geom_horizontal;Vector *face_geom_volume;} ProblemData;

#define VectorGrid(v) ((v)->grid)
#define GridSubgrid(g,i) ((g)->local->items[i])
#define ProblemDataZmult(p) ((p)->zmult)
#define SubgridIX(s) ((s)->ix)
#define SubgridIY(s) ((s)->iy)
#define SubgridIZ(s) ((s)->iz)
#define SubgridNZ(s) ((s)->nz)
#define SubgridDX(s) background.dx
#define SubgridDY(s) background.dy
#define SubgridDZ(s) background.dz
#define BackgroundY(b) ((b)->y)
#define VectorUpdateGodunov 4
typedef Vector VectorUpdateCommHandle;
Vector *InitVectorUpdate(Vector *v,int mode){
 (void)mode;
 /* Simulate local-block halo exchange; real MPI communication is not tested. */
 for(int a=0;a<v->grid->local->size;a++){
 Subvector *s=v->s[a];
 for(int k=s->box.iz;k<s->box.iz+s->box.nz;k++)
 for(int j=s->box.iy;j<s->box.iy+s->box.ny;j++)
 for(int i=s->box.ix;i<s->box.ix+s->box.nx;i++)
 for(int b=0;b<v->grid->local->size;b++){
 Subgrid *g=v->grid->local->items[b];
 if(i>=g->ix && i<g->ix+g->nx && j>=g->iy && j<g->iy+g->ny && k>=g->iz && k<g->iz+g->nz)
 SubvectorData(s)[SubvectorEltIndex(s,i,j,k)]=SubvectorData(v->s[b])[SubvectorEltIndex(v->s[b],i,j,k)];
 }}return v;}
void FinalizeVectorUpdate(Vector *v){(void)v;}
'''
MAIN = r'''
int main(int argc,char **argv){
 assert(argc==4);int ones=atoi(argv[3]);int checks=0;
 for(int split=0;split<3;split++){
 SubgridArray *a=NewSubgridArray();
 if(split==0){Subgrid s={0,0,0,2,3,2};AppendSubgrid(DuplicateSubgrid(&s),a);}
 if(split==1){Subgrid s={2,1,0,2,2,2};AppendSubgrid(DuplicateSubgrid(&s),a);}
 if(split==2){Subgrid s={0,0,0,2,3,2},t={2,0,0,2,3,2};AppendSubgrid(DuplicateSubgrid(&s),a);AppendSubgrid(DuplicateSubgrid(&t),a);}
 Grid *g=NewGrid(a,a);ProblemData pd={0};CreateComputePkgs(g);
 pd.face_dx_scale=ReadFaceMetric(g,argv[1],1);
 pd.face_dy_scale=ReadFaceMetric(g,argv[2],0);
 pd.zmult=NewVectorType(g,1,1,0);
 for(int is=0;is<a->size;is++){
 Subvector *zs=pd.zmult->s[is];
 for(int k=zs->box.iz;k<zs->box.iz+zs->box.nz;k++)
 for(int j=zs->box.iy;j<zs->box.iy+zs->box.ny;j++)
 for(int i=zs->box.ix;i<zs->box.ix+zs->box.nx;i++)
 SubvectorData(zs)[SubvectorEltIndex(zs,i,j,k)]=.5+.1*k;
 }
 PrepareFaceGeometryCache(&pd);
 Vector *cache[8]={pd.face_geom_dx,pd.face_geom_dy,pd.face_geom_west,pd.face_geom_east,pd.face_geom_south,pd.face_geom_north,pd.face_geom_horizontal,pd.face_geom_volume};
 for(int is=0;is<a->size;is++){
 Subgrid *s=a->items[is];Subvector *idx_sub=cache[0]->s[is];
 const double *data[8];for(int n=0;n<8;n++)data[n]=SubvectorData(cache[n]->s[is]);
 for(int k=s->iz;k<s->iz+s->nz;k++)for(int j=s->iy;j<s->iy+s->ny;j++)for(int i=s->ix;i<s->ix+s->nx;i++){
 double xs=ones?1:1+.1*j+.01*k,xn=ones?1:1+.1*(j+1)+.01*k;
 double yw=ones?1:1+.03*i,ye=ones?1:1+.03*(i+1);
 double dx=50*(xs+xn),dy=100*(yw+ye),t=20*(.5+.1*k);
 double expected[8]={dx,dy,200*yw*t,200*ye*t,100*xs*t,100*xn*t,dx*dy,dx*dy*t};
 int idx=SubvectorEltIndex(idx_sub,i,j,k);
 assert(idx!=SubvectorEltIndex(pd.zmult->s[is],i,j,k));
 for(int n=0;n<8;n++){
 assert(idx==SubvectorEltIndex(cache[n]->s[is],i,j,k));
 assert(fabs(data[n][idx]-expected[n])<1e-12*fmax(1,fabs(expected[n])));checks++;}
 Subvector *source=pd.face_dx_scale->s[is];
 assert(isfinite(SubvectorData(source)[SubvectorEltIndex(source,i-1,j-1,k-1)]));
 }
 }
 if(split==2)for(int n=0;n<8;n++){
 Subvector *s=cache[n]->s[0],*t=cache[n]->s[1];
 assert(SubvectorData(s)[SubvectorEltIndex(s,2,1,1)]==SubvectorData(t)[SubvectorEltIndex(t,2,1,1)]);checks++;}

 }
 printf("%d face reader, shared-index and initialized-cache checks passed\n",checks);return 0;
}
'''



def write_pfb(path, nx, ny, nz, value, refinement=1):
    # Multiple x subgrids exercise general whole-file PFB layout.
    with open(path, "wb") as f:
        f.write(struct.pack(">3d3i3di", 0, 0, 0, nx, ny, nz, 100, 200, 10, 2))
        for start, width in ((0, 2), (2, nx - 2)):
            f.write(struct.pack(">9i", start, 0, 0, width, ny, nz, refinement, refinement, refinement))
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
        cache = source[source.index("static void PrepareFaceGeometryCache"):
                       source.index("/*--------------------------------------------------------------------------\n * SetProblemData", source.index("static void PrepareFaceGeometryCache"))]
        (d / "test.c").write_text(SHIMS + reader + cache + MAIN)
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-fsanitize=address,undefined",
                        "-I" + str(ROOT / "pfsimulator/parflow_lib"),
                        str(d / "test.c"), "-lm", "-o", str(d / "test")], check=True)
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0")
        for ones in (True, False):
            write_pfb(d / "dx.pfb", 4, 4, 2, lambda i, j, k: 1 if ones else 1 + .1*j + .01*k)
            write_pfb(d / "dy.pfb", 5, 3, 2, lambda i, j, k: 1 if ones else 1 + .03*i)
            subprocess.run([str(d / "test"), str(d / "dx.pfb"), str(d / "dy.pfb"), str(int(ones))], check=True, env=env)
        # Native C output may use zero in these header fields.
        write_pfb(d / "dx.pfb", 4, 4, 2, lambda *args: 1, refinement=0)
        write_pfb(d / "dy.pfb", 5, 3, 2, lambda *args: 1, refinement=0)
        subprocess.run([str(d / "test"), str(d / "dx.pfb"), str(d / "dy.pfb"), "1"], check=True, env=env)
        write_pfb(d / "dx.pfb", 4, 4, 2, lambda *args: 1, refinement=2)
        result = subprocess.run([str(d / "test"), str(d / "dx.pfb"), str(d / "dy.pfb"), "1"], env=env, capture_output=True)
        assert result.returncode == 2, result.stderr
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
