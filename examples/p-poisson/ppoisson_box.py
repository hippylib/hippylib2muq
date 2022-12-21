import os
import sys
import math
import yaml

#  import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import dolfin as dl
from hippylib import *

import muq.Modeling as mm
import muq.SamplingAlgorithms as ms
import hippylib2muq as hm

from nonlinearPPoissonProblem import *

# np.random.seed(seed=1)


def true_model(prior):
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    parRandom.normal(1.0, noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise, mtrue)
    return mtrue


def export2XDMF(x, Vh, fid):
    fid.parameters["functions_share_mesh"] = True
    fid.parameters["rewrite_function_mesh"] = False

    fun = vector2Function(x, Vh)
    fid.write(fun, 0)


def Boundary(x, on_boundary):
    return on_boundary


class BottomBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[2], 0)


class SideBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            dl.near(x[0], 0)
            or dl.near(x[0], Length)
            or dl.near(x[1], 0)
            or dl.near(x[1], Width)
        )


class TopBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[2], Height)


def generate_MHoptions():
    opts = dict()
    opts["BurnIn"] = inargs["nsample"] // 10
    opts["NumSamples"] = inargs["nsample"] + opts["BurnIn"]
    opts["PrintLevel"] = 2
    opts["Beta"] = inargs["beta"]
    opts["StepSize"] = inargs["tau"]
    return opts


def setup_proposal(propName, options, problem, propGaussian):
    if propName == "pcn":
        proposal = ms.CrankNicolsonProposal(options, problem, propGaussian)
    return proposal


def setup_kernel(kernName, options, problem, proposal):
    if kernName == "mh":
        kernel = ms.MHKernel(options, problem, proposal)

    return kernel


def generate_starting():
    noise = dl.Vector()
    nu.init_vector(noise, "noise")
    parRandom.normal(1.0, noise)
    pr_s = model.generate_vector(PARAMETER)
    post_s = model.generate_vector(PARAMETER)
    nu.sample(noise, pr_s, post_s, add_mean=True)
    x0 = hm.dlVector2npArray(post_s)
    return x0


def run_MCMC(options, kernel, startpoint):
    # Construct the MCMC sampler
    sampler = ms.SingleChainMCMC(options, [kernel])

    # Run the MCMC sampler
    samples = sampler.Run([startpoint])

    if "AcceptanceRate" in dir(kernel):
        return samples, kernel.AcceptanceRate(), sampler.TotalTime()
    elif "AcceptanceRates" in dir(kernel):
        return samples, kernel.AcceptanceRates(), sampler.TotalTime()


def data_file(action, target=None, data=None):
    f = h5py.File("data.h5", action)
    if action == "w":
        f["/target"] = target
        f["/data"] = data

        f.close()
        return

    elif action == "r":
        target = f["/target"][...]
        data = f["/data"][...]

        f.close()

        return target, data


class ExtractBottomData:
    def __init__(self, mesh, Vh):
        bmesh = dl.BoundaryMesh(mesh, "exterior")
        bmarker = dl.MeshFunction("size_t", mesh, bmesh.topology().dim())
        for c in dl.cells(bmesh):
            if math.isclose(c.midpoint().z(), 0):
                bmarker[c] = 1

        smesh = dl.SubMesh(bmesh, bmarker, 1)

        self.vertex_s2b = smesh.data().array("parent_vertex_indices", 0)
        self.vertex_b2p = bmesh.entity_map(0).array()
        self.vertex2dof = dl.vertex_to_dof_map(Vh)
        self.coordinates = smesh.coordinates()

        self.tria = tri.Triangulation(
            self.coordinates[:, 0], self.coordinates[:, 1], smesh.cells()
        )

    def get_dim(self):
        return self.coordinates.shape[0]

    def get_bottom_data(self, arr):
        return arr[self.vertex2dof[self.vertex_b2p[self.vertex_s2b]]]

    def plot_array(self, arr, vmin=None, vmax=None, cmap=None, fname=None):
        val = arr[self.vertex2dof[self.vertex_b2p[self.vertex_s2b]]]

        if vmax is None:
            vmax = np.max(val)
        if vmin is None:
            vmin = np.min(val)

        plt.tripcolor(self.tria, val, shading="gouraud", vmin=vmin, vmax=vmax)
        if cmap:
            plt.set_cmap(cmap)

        plt.axis("off")
        plt.gca().set_aspect("equal")

        if fname:
            plt.savefig(fname, dpi=100, bbox_inches="tight", pad_inches=0)

        plt.show()


class TracerBottomFlux:
    def __init__(self, n):
        self.m = pde.generate_parameter()
        self.mf = vector2Function(self.m, Vh[PARAMETER])
        self.f = self.mf * ds(1)

        self.tracer = QoiTracer(n)
        self.ct = 0

    def update_tracer(self, sample):
        self.mf.vector().set_local(sample)
        y = dl.assemble(self.f)
        self.tracer.append(self.ct, y)
        self.ct += 1


class TracerSideFlux:
    def __init__(self, ds, p, n):
        self.n = dl.FacetNormal(mesh)
        self.ds = ds
        self.p = p

        self.tracer = QoiTracer(n)
        self.ct = 0

    def form(self, u):
        grad_u = dl.nabla_grad(u)
        etah = dl.inner(grad_u, grad_u)

        return etah ** (0.5 * (self.p - 2)) * dl.dot(grad_u, self.n) * self.ds

    def eval(self, u):
        uf = vector2Function(u, Vh[STATE])
        return dl.assemble(self.form(uf))

    def update_tracer(self, state):
        y = self.eval(state)
        self.tracer.append(self.ct, y)
        self.ct += 1


def paramcoord2eigencoord(V, B, x):
    """
    Projection a parameter vector to eigenvector.

    y = V^T * B * x

    :param V multivector: eigenvectors
    :param operator: the right-hand side operator in the generalized eig problem
    :param x np.array: parameter data
    """
    # convert np.array to multivector
    nvec = 1
    Xvecs = MultiVector(model.generate_vector(PARAMETER), nvec)
    hm.npArray2dlVector(x, Xvecs[0])

    # multipy B
    BX = MultiVector(Xvecs[0], nvec)
    MatMvMult(B, Xvecs, BX)
    VtBX = BX.dot_mv(V)

    return VtBX.transpose()


# def check_positiveness_parm2obs(nsamps: int):
#     noise = dl.Vector()
#     prior.init_vector(noise, "noise")
#     m = dl.Vector()
#     prior.init_vector(m, 0)
#     u = pde.generate_state()
#     Bu = dl.Vector(misfit.B.mpi_comm())
#     misfit.B.init_vector(Bu, 0)
#
#     for i in range(nsamps):
#         parRandom.normal(1.0, noise)
#         prior.sample(noise, m)
#
#         x = [u, m, None]
#
#         pde.solveFwd(x[STATE], x)
#         misfit.B.mult(x[STATE], Bu)
#
#         if (Bu[:] < 0.).any():
#             print("Negative values of Bu encountered.")


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + "/ppoisson.yaml") as fid:
        inargs = yaml.full_load(fid)

    sep = "\n" + "#" * 80 + "\n"

    #
    #  Set up the mesh and finite element spacs
    #
    ndim = 3
    Length = 1.0
    Width = Length
    Height = 0.05

    nx = inargs["nelement"][0]
    ny = nx
    nz = inargs["nelement"][1]

    mesh = dl.BoxMesh(dl.Point(0, 0, 0), dl.Point(Length, Width, Height), nx, ny, nz)
    bottom = BottomBoundary()
    # side = SideBoundary()
    # top = TopBoundary()

    # mesh for the parameter field
    mesh_param = dl.RectangleMesh(dl.Point(0, 0), dl.Point(Length, Width), nx, ny)

    rank = dl.MPI.rank(mesh.mpi_comm())

    Vh1 = dl.FunctionSpace(mesh_param, "Lagrange", 1)
    Vh2 = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh = [Vh2, Vh1, Vh2]

    if rank == 0:
        print(
            "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
                Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()
            )
        )

    # extract_bottom = ExtractBottomData(mesh, Vh[PARAMETER])

    #
    #  Set up the forward problem
    #
    dl.parameters["form_compiler"]["quadrature_degree"] = 3

    # bc = dl.DirichletBC(Vh[STATE], dl.Constant(0.0), side)

    #  Bottom and side boundary markers
    boundary_markers = dl.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundary_markers.set_all(0)

    bottom.mark(boundary_markers, 1)
    # side.mark(boundary_markers, 2)
    ds = dl.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

    order_ppoisson = 3.0
    f = dl.Constant(1.0)
    functional = NonlinearPPossionForm(order_ppoisson, f, ds(1))
    pde = EnergyFunctionalPDEVariationalProblem(Vh, functional, [], [])

    pde.solver = dl.PETScKrylovSolver("cg", "icc")  # amg_method())
    pde.solver_fwd_inc = dl.PETScKrylovSolver("cg", "icc")  # amg_method())
    pde.solver_adj_inc = dl.PETScKrylovSolver("cg", "icc")  # amg_method())
    pde.fwd_solver.solver = dl.PETScKrylovSolver("cg", "icc")  # amg_method())

    # pde.fwd_solver.parameters["print_level"] = 1
    pde.fwd_solver.parameters["gdu_tolerance"] = 1e-14
    pde.fwd_solver.parameters["LS"]["max_backtracking_iter"] = 20

    pde.solver_fwd_inc.parameters = pde.solver.parameters
    pde.solver_adj_inc.parameters = pde.solver.parameters
    pde.fwd_solver.solver.parameters = pde.solver.parameters

    #
    #  Set up the prior
    #
    # gamma = 1.0
    # delta = 1.0
    gamma = 0.1
    delta = 0.5

    theta0 = 2.0
    theta1 = 0.5
    alpha = math.pi / 4
    anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree=1)
    anis_diff.set(theta0, theta1, alpha)

    prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, anis_diff, robin_bc=True)
    if rank == 0:
        print(
            "Prior regularization: (delta_x - gamma*Laplacian)^order: "
            "delta={0}, gamma={1}, order={2}".format(delta, gamma, 2)
        )
        print("Correlatio length: ", math.sqrt(gamma/delta))
        print("Variace: ", 1./(gamma*delta))
        print("")

    mtrue0 = true_model(prior)

    if inargs["savefig"]:
        pp = nb.plot(
            dl.Function(Vh[PARAMETER], mtrue0), colorbar=False, cmap="coolwarm"
        )
        plt.gca().set_aspect("equal")
        # plt.show()
        plt.savefig("mtrue.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    mtrue = pde.parameter_projection(mtrue0)

    # ##########################################################
    # # with dl.XDMFFile(mesh.mpi_comm(), 'mtrue.xdmf') as fid:
    # #     export2XDMF(mtrue, Vh[PARAMETER], fid)
    # #
    # # marr = mtrue.get_local()
    # # extract_bottom.plot_array(marr, fname='mtrue.png')
    # ##########################################################

    #
    #  Set up the misfit functional and generate synthetic observations
    #
    ntargets = 300
    Mpar = 1e4

    if inargs["write_data"]:
        eps = 0.05
        dummy1 = np.random.uniform(
            Length * (0.0 + eps), Length * (1.0 - eps), [ntargets, 1]
        )
        dummy2 = np.random.uniform(
            Width * (0.0 + eps), Width * (1.0 - eps), [ntargets, 1]
        )
        dummy3 = np.full((ntargets, 1), Height)
        targets = np.concatenate([dummy1, dummy2, dummy3], axis=1)

        if rank == 0:
            print("Number of observation points: {0}".format(ntargets))

        misfit = MultPointwiseStateObservation(Vh[STATE], targets, Mpar)

        # True state

        utrue = pde.generate_state()
        x = [utrue, mtrue, None]
        pde.solveFwd(x[STATE], x)

        misfit.B.mult(x[STATE], misfit.d)
        y_true = misfit.d.get_local()
        parRandom.speckle(Mpar, misfit.d)
        y_noise = misfit.d.get_local()
        # print("Max error: ", np.linalg.norm((y_noise - y_true) / y_true, ord=np.inf))

        data_file("w", target=targets, data=misfit.d.get_local())

        ##########################################################
        # with dl.XDMFFile(mesh.mpi_comm(), 'utrue.xdmf') as fid:
        #     export2XDMF(utrue, Vh[STATE], fid)
        ##########################################################

    else:
        targets, data = data_file("r")

        misfit = MultPointwiseStateObservation(Vh[STATE], targets, Mpar)
        misfit.d.set_local(data)

    # Check positiveness of parameter to observable map
    # pde.fwd_solver.parameters["print_level"] = 1
    # pde.fwd_solver.parameters["LS"]["max_backtracking_iter"] = 100
    # pde.fwd_solver.parameters["gdu_tolerance"] = 1e-16
    # check_positiveness_parm2obs(1000)

    model = MixedDimensionModel(pde, prior, misfit)

    #
    # Gradient check
    #
    # m0 = dl.interpolate(dl.Expression("sin(x[0])", degree=5), Vh[PARAMETER])
    # _ = modelVerify(model, m0.vector())
    # plt.show()

    #
    #  Compute the MAP point
    #
    m = prior.mean.copy()
    solver = ReducedSpaceNewtonCG(model)
    # solver.parameters["rel_tolerance"] = 1e-8
    # solver.parameters["abs_tolerance"] = 1e-12
    solver.parameters["max_iter"] = 25
    solver.parameters["GN_iter"] = 5
    solver.parameters["globalization"] = "LS"
    solver.parameters["LS"]["c_armijo"] = 1e-4

    x = solver.solve([None, m, None])

    if rank == 0:
        if solver.converged:
            print("\nConverged in ", solver.it, " iterations.")
        else:
            print("\nNot Converged")

        print("Termination reason:  ", solver.termination_reasons[solver.reason])
        print("Final gradient norm: ", solver.final_grad_norm)
        print("Final cost:          ", solver.final_cost)

    if inargs["savefig"]:
        pp = nb.plot(dl.Function(Vh[PARAMETER], m), colorbar=False, cmap="coolwarm")
        plt.gca().set_aspect("equal")
        # plt.show()
        plt.savefig("map.png", bbox_inches="tight", pad_inches=0)
        plt.close()

    ##########################################################
    # with dl.XDMFFile(mesh.mpi_comm(), "map.xdmf") as fid:
    #     export2XDMF(x[PARAMETER], Vh[PARAMETER], fid)
    #
    #  marr = x[PARAMETER].get_local()
    #  extract_bottom.plot_array(marr, fname='map.png')
    ##########################################################
    #  ##########################################################
    #  with dl.XDMFFile(mesh.mpi_comm(), 'map_state.xdmf') as fid:
    #      export2XDMF(x[STATE], Vh[STATE], fid)
    #  ###########################################################

    #
    #  Compute the low-rank based Laplace approximation of the posterior
    #
    pde.nit = 0
    model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
    Hmisfit = ReducedHessian(model, misfit_only=True)
    k = 100
    p = 20

    if rank == 0:
        print(
            "Single/Double Pass Algorithm. Requested eigenvectors: "
            "{0}; Oversampling {1}.".format(k, p)
        )

    Omega = MultiVector(x[PARAMETER], k + p)
    parRandom.normal(1.0, Omega)
    lmbda, V = doublePassG(Hmisfit, prior.R, prior.Rsolver, Omega, k)

    if inargs["savefig"]:
        plt.plot(range(0, k), lmbda, "b*", range(0, k + 1), np.ones(k + 1), "-r")
        plt.yscale("log")
        plt.xlabel("number")
        plt.ylabel("eigenvalue")
        plt.show()

    nu = GaussianLRPosterior(prior, lmbda, V)
    nu.mean = x[PARAMETER]
    # #
    # # #  ###########################################################
    # # #  print("Prior and LA-posterior pointwise variance fields")
    # # #  nsamples = 5
    # # #  noise = dl.Vector()
    # # #  nu.init_vector(noise, "noise")
    # # #  sprior = dl.Function(Vh[PARAMETER], name="sample_prior")
    # # #  spost = dl.Function(Vh[PARAMETER], name="sample_post")
    # # #
    # # #  for i in range(nsamples):
    # # #      parRandom.normal(1., noise)
    # # #      nu.sample(noise, sprior.vector(), spost.vector())
    # # #      extract_bottom.plot_array(sprior.vector().get_local())#, vmax=1., vmin=-1.)
    # # #      extract_bottom.plot_array(spost.vector().get_local())#, vmax=1., vmin=-1.)
    # # #
    # # #      #  print(sprior.vector().get_local().max(), sprior.vector().get_local().min())
    # #
    # # #  ###########################################################

    #
    #  Set up ModPieces for implementing MCMC methods
    #
    idparam = mm.IdentityOperator(Vh[PARAMETER].dim())

    # log Gaussian Prior ModPiece
    gaussprior = hm.BiLaplaceGaussian(prior)
    log_gaussprior = gaussprior.AsDensity()

    # parameter to log likelihood Modpiece
    param2likelihood = hm.Param2LogLikelihood(model)

    # log target ModPiece
    log_target = mm.DensityProduct(2)

    workgraph = mm.WorkGraph()

    # Identity operator for the parameters
    workgraph.AddNode(idparam, "Identity")

    # Prior model
    workgraph.AddNode(log_gaussprior, "Prior")

    # Likelihood model
    workgraph.AddNode(param2likelihood, "Likelihood")

    # Posterior
    workgraph.AddNode(log_target, "Target")

    workgraph.AddEdge("Identity", 0, "Prior", 0)
    workgraph.AddEdge("Prior", 0, "Target", 0)

    workgraph.AddEdge("Identity", 0, "Likelihood", 0)
    workgraph.AddEdge("Likelihood", 0, "Target", 1)

    # Enable caching
    if inargs["method"] not in ("hpcn", ""):
        log_gaussprior.EnableCache()
        param2likelihood.EnableCache()

    # Construct the problem
    postDens = workgraph.CreateModPiece("Target")
    problem = ms.SamplingProblem(postDens)

    #
    #  Exploring the posterior via MCMC methods
    #
    opts = generate_MHoptions()

    if inargs["method"] == "hpcn":
        gaussprop = hm.LAPosteriorGaussian(nu)
        propName = "pcn"
        kernName = "mh"
    else:
        sys.exit()

    propos = setup_proposal(propName, opts, problem, gaussprop)
    kern = setup_kernel(kernName, opts, problem, propos)

    if inargs["fname"] != "":
        fid = h5py.File(inargs["fname"] + ".h5", "w")

        fid["/"].attrs["Method"] = inargs["method"]
        fid["/"].attrs["Beta"] = opts["Beta"]
        fid["/"].attrs["StepSize"] = opts["StepSize"]
    else:
        fid = None

    # Projection matrix from fe coordinates to eigen coordinates
    num_eigcoord = 25
    TT = MultiVector(model.generate_vector(PARAMETER), num_eigcoord)
    for i in range(num_eigcoord):
        # TODO: is there a better way to to this??
        TT[i].axpy(1.0, V[i])

    if inargs["fname"]:
        x0 = generate_starting()

        samps, acceptrate, etime = run_MCMC(opts, kern, x0)

        if isinstance(fid, h5py.File):
            samps_data = np.zeros((num_eigcoord, samps.size()))

            for j in range(samps.size()):
                xx = samps[j].state[0]

                samps_data[:, j] = paramcoord2eigencoord(TT, prior.R, xx).reshape(-1)

            sname = "sample"
            sampsDataset = fid.create_dataset(
                sname, (num_eigcoord, samps.size()), dtype=np.float
            )

            sampsDataset.attrs["AR"] = acceptrate
            sampsDataset.attrs["etime"] = etime
            sampsDataset[...] = samps_data

        del samps

        fid["/"].attrs["neval"] = param2likelihood.GetNumCalls("Evaluate")
        fid.close()
