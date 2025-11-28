import numpy as np
from abc import ABC, abstractmethod
from time import perf_counter, sleep
import threading

import vtk
import pyvista
from .rod import CircularCrossSection

from tdcrobots.rods._base import CosseratRod_PetrovGalerkin


from tdcrobots.interactions import nPointInteraction


class _VisualTwinBase(ABC):
    def __init__(self, contr):
        self.contr = contr
        self.actors = []
        if not hasattr(contr, "visual_twins"):
            contr.visual_twins = [self]
        else:
            contr.visual_twins.append(self)

    @abstractmethod
    def update_state(self, t, q, u):
        pass


class VisualTendon(_VisualTwinBase):
    def __init__(
        self, tendon: nPointInteraction, r_tendon=1e-3, color=(255, 255, 255), opacity=1
    ):
        super().__init__(tendon)
        poly_data = vtk.vtkPolyData()
        # points
        npts = 2
        ncon = len(self.contr.connectivity)
        self.vtkpoints = vtk.vtkPoints()
        self.vtkpoints.SetNumberOfPoints(npts * ncon)
        poly_data.SetPoints(self.vtkpoints)

        # cells
        poly_data.Allocate(ncon)
        for i in range(ncon):
            poly_data.InsertNextCell(
                vtk.VTK_LINE, npts, list(range(i * npts, (i + 1) * npts))
            )
        filter = vtk.vtkTubeFilter()
        filter.SetRadius(r_tendon)
        filter.SetInputData(poly_data)
        filter.SetNumberOfSides(50)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(filter.GetOutputPort())
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputData(poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(([c / 255 for c in color]))
        actor.GetProperty().SetOpacity(opacity)
        self.actors.append(actor)

    def update_state(self, t, q, u):
        points = []
        for j, k in self.contr.connectivity:
            points.append(self.contr.r_OPk(t, q, j))
            points.append(self.contr.r_OPk(t, q, k))
        for i, p in enumerate(points):
            self.vtkpoints.SetPoint(i, p)
        self.vtkpoints.Modified()



class VisualRodBody(_VisualTwinBase):
    def __init__(
        self, rod, nelement_visual=1, subdivision=3, color=(82, 108, 164), opacity=1
    ):
        super().__init__(rod)
        self.rod = rod
        self.nelement_visual = nelement_visual

        if isinstance(rod.cross_section, CircularCrossSection):
            weights = [
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
            ]
            degrees = [2, 2, 1]
            ctype = vtk.VTK_BEZIER_WEDGE
        # elif isinstance(rod.cross_section, RectangularCrossSection):
        #     npts = 16
        #     weights = [1] * 16
        #     degrees = [1, 1, 3]
        #     ctype = vtk.VTK_BEZIER_HEXAHEDRON
        else:
            raise NotImplementedError

        ugrid = vtk.vtkUnstructuredGrid()

        # points
        self.body_points = vtk.vtkPoints()
        self.body_points.SetNumberOfPoints(6 * (nelement_visual + 1))
        ugrid.SetPoints(self.body_points)

        # cells
        ugrid.Allocate(nelement_visual)
        for i in range(nelement_visual):
            ugrid.InsertNextCell(
                ctype,
                12,
                list(range(i * 6, i * 6 + 3))
                + list(range((i + 1) * 6, (i + 1) * 6 + 3))
                + list(range(i * 6 + 3, (i + 1) * 6))
                + list(range((i + 1) * 6 + 3, (i + 2) * 6)),
            )

        # point data
        pdata = ugrid.GetPointData()
        value = weights * (nelement_visual + 1)
        parray = vtk.vtkDoubleArray()
        parray.SetName("RationalWeights")
        parray.SetNumberOfTuples(6)
        parray.SetNumberOfComponents(1)
        for i, vi in enumerate(value):
            parray.InsertTuple(i, [vi])
        pdata.SetRationalWeights(parray)

        # cell data
        cdata = ugrid.GetCellData()
        carray = vtk.vtkIntArray()
        carray.SetName("HigherOrderDegrees")
        carray.SetNumberOfTuples(nelement_visual)
        carray.SetNumberOfComponents(3)
        for i in range(nelement_visual):
            carray.InsertTuple(i, degrees)
        cdata.SetHigherOrderDegrees(carray)

        filter = vtk.vtkDataSetSurfaceFilter()
        filter.SetInputData(ugrid)
        filter.SetNonlinearSubdivisionLevel(subdivision)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor([c / 255 for c in color])
        actor.GetProperty().SetOpacity(opacity)
        self.actors.append(actor)
        self.xis_visual = np.linspace(0.0, 1.0, self.nelement_visual + 1)
        basis_r = np.array([rod.basis_functions_r(xi) for xi in self.xis_visual])
        # TODO: the pre-evaluation of N and N_xi is only because slow basis_functions_r.
        # need to speed up basis_functions_r in Cardillo,
        # then use r_OP and A_IB directly in update_state function
        self.Ns = basis_r[:, 0, :]
        self.N_xis = basis_r[:, 1, :]
        # control points on circle
        phis = np.linspace(0.0, 2.0 * np.pi, 3, endpoint=False)
        xys1 = (
            np.stack([np.zeros_like(phis), np.cos(phis), np.sin(phis)], axis=1)
            * rod.cross_section.radius
        )
        # control points out of circle
        phis2 = phis + (np.pi / 3.0)
        xys2 = np.stack([np.zeros_like(phis), np.cos(phis2), np.sin(phis2)], axis=1) * (
            2.0 * rod.cross_section.radius
        )
        self.control_pts_circle = np.concatenate([xys1, xys2], axis=0)

    def update_state(self, t, q, u):
        rod = self.rod
        nxi = self.nelement_visual + 1
        control_pts = np.empty((nxi * 6, 3), dtype=np.float64)
        for i, xi, N, N_xi in zip(range(nxi), self.xis_visual, self.Ns, self.N_xis):
            el = 1
            qe = q
            r_OC, A_IB, _, _ = self.rod._eval(qe, xi, N, N_xi)
            pts = r_OC + self.control_pts_circle @ A_IB.T
            base = i * 6
            control_pts[base : base + 6] = pts
        body_points = self.body_points
        set_point = body_points.SetPoint
        for i, p in enumerate(control_pts):
            set_point(i, p)
        body_points.Modified()


class _VisualvtkSource(_VisualTwinBase):
    def __init__(
        self,
        contr,
        xi,
    ):
        super().__init__(contr)
        self.xi = xi
        self.H_IB = vtk.vtkMatrix4x4()
        self.H_IB.Identity()
        if isinstance(self.contr, CosseratRod_PetrovGalerkin):
            self.N, self.N_xi = self.contr.basis_functions_r(xi)

    def add_vtk_source(
        self,
        source,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        color=(255, 255, 255),
        opacity=1,
    ):

        H_BM = np.block(
            [
                [A_BM, B_r_CP[:, None]],
                [0, 0, 0, 1],
            ]
        )
        _H_IB = vtk.vtkMatrixToLinearTransform()
        _H_IB.SetInput(self.H_IB)
        _H_IM = vtk.vtkTransform()
        _H_IM.PostMultiply()
        _H_IM.SetMatrix(H_BM.flatten())
        _H_IM.Concatenate(_H_IB)
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputConnection(source.GetOutputPort())
        tf_filter.SetTransform(_H_IM)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor([c / 255 for c in color])
        actor.GetProperty().SetOpacity(opacity)
        self.actors.append(actor)

    def update_state(self, t, q, u):
        xi = self.xi
        A_IB = self.contr.A_IB(t, q, xi)
        r_OP = self.contr.r_OP(t, q, xi)
        for i in range(3):
            for j in range(3):
                self.H_IB.SetElement(i, j, A_IB[i, j])
            self.H_IB.SetElement(i, 3, r_OP[i])


class VisualSTL(_VisualvtkSource):
    def __init__(
        self,
        contr,
        stl_file,
        xi=None,
        scale=1e-3,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        color=(255, 255, 255),
        opacity=1,
    ):
        super().__init__(contr, xi)
        source = vtk.vtkSTLReader()
        source.SetFileName(stl_file)
        source.Update()
        self.add_vtk_source(source, A_BM * scale, B_r_CP, color, opacity)


class VisualCoordSystem(_VisualvtkSource):
    def __init__(
        self,
        contr,
        length,
        xi=None,
        resolution=30,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        opacity=1,
    ):
        super().__init__(contr, xi)
        source = vtk.vtkArrowSource()
        source.SetTipResolution(resolution)
        source.SetShaftResolution(resolution)
        for i in range(3):
            if i == 0:
                color = (255, 0, 0)
            elif i == 1:
                A_BM = A_BM @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                color = (0, 255, 0)
            elif i == 2:
                A_BM = A_BM @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
                color = (0, 0, 255)
            self.add_vtk_source(source, A_BM * length, B_r_CP, color, opacity)


class VisualArUco(_VisualTwinBase):
    def __init__(
        self,
        contr,
        xi=None,
        mk_size=0.04,
        mk_dis=0.045,
        A_BM=np.eye(3),
        B_r_CP=np.zeros(3),
        opacity=1,
    ):
        super().__init__(contr)
        self.xi = xi
        if isinstance(self.contr, CosseratRod_PetrovGalerkin):
            self.N, self.N_xi = self.contr.basis_functions_r(xi)
        from cv2 import aruco

        n_row = 2
        n_col = 2
        x0 = -mk_size / 2 - mk_dis / 2
        y0 = -x0
        h0 = 1e-4
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        quads_black = vtk.vtkCellArray()
        quads_white = vtk.vtkCellArray()
        points = vtk.vtkPoints()
        for row in range(n_row):
            for col in range(n_col):
                id = row * n_col + col
                qrcode = aruco_dict.generateImageMarker(id, aruco_dict.markerSize + 2)
                bit_size = mk_size / (aruco_dict.markerSize + 2)

                # Create a triangle
                n_bits = qrcode.shape[0]
                for i in range(n_bits + 1):
                    for j in range(n_bits + 1):
                        points.InsertNextPoint(
                            x0 + col * mk_dis + j * bit_size,
                            y0 - row * mk_dis - i * bit_size,
                            h0,
                        )

                for i in range(n_bits):
                    for j in range(n_bits):
                        quad = vtk.vtkQuad()
                        quad.GetPointIds().SetId(
                            0, id * (n_bits + 1) ** 2 + i * (n_bits + 1) + j
                        )
                        quad.GetPointIds().SetId(
                            1, id * (n_bits + 1) ** 2 + (i + 1) * (n_bits + 1) + j
                        )
                        quad.GetPointIds().SetId(
                            2, id * (n_bits + 1) ** 2 + (i + 1) * (n_bits + 1) + j + 1
                        )
                        quad.GetPointIds().SetId(
                            3, id * (n_bits + 1) ** 2 + i * (n_bits + 1) + j + 1
                        )
                        if qrcode[i, j] == 0:
                            quads_black.InsertNextCell(quad)
                        else:
                            quads_white.InsertNextCell(quad)

        self.H_IB = vtk.vtkMatrix4x4()
        self.H_IB.Identity()
        H_BM = np.block(
            [
                [A_BM, B_r_CP[:, None]],
                [0, 0, 0, 1],
            ]
        )
        _H_IB = vtk.vtkMatrixToLinearTransform()
        _H_IB.SetInput(self.H_IB)
        _H_IM = vtk.vtkTransform()
        _H_IM.PostMultiply()
        _H_IM.SetMatrix(H_BM.flatten())
        _H_IM.Concatenate(_H_IB)

        # qrcode
        for triangles, color in zip(
            [quads_black, quads_white], [(0, 0, 0), (255, 255, 255)]
        ):
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(triangles)

            filter = vtk.vtkTransformPolyDataFilter()
            filter.SetInputData(polydata)
            filter.SetTransform(_H_IM)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(filter.GetOutputPort())
            actor = vtk.vtkActor()
            actor.GetProperty().SetColor([c / 255 for c in color])
            actor.GetProperty().SetOpacity(opacity)
            actor.SetMapper(mapper)
            self.actors.append(actor)
            # subsystem.appendfilter.AddInputConnection(filter.GetOutputPort())

    def update_state(self, t, q, u):
        xi = self.xi
        A_IB = self.contr.A_IB(t, q, xi)
        r_OP = self.contr.r_OP(t, q, xi)
        for i in range(3):
            for j in range(3):
                self.H_IB.SetElement(i, j, A_IB[i, j])
            self.H_IB.SetElement(i, 3, r_OP[i])


class BackgroundPlotter(pyvista.Plotter):
    def __init__(self, system, window_size, **kwargs):
        super().__init__(window_size=window_size, **kwargs)
        self.__visual_twins = []
        self.do_render = False
        self.system = system
        for contr in system.contributions:
            if hasattr(contr, "visual_twins"):
                for twin in contr.visual_twins:
                    self.__add_visual_twin(twin)
            if hasattr(contr, "rod_nodes"):
                for node in contr.rod_nodes:
                    if hasattr(node, "visual_twins"):
                        for twin in node.visual_twins:
                            self.__add_visual_twin(twin)

        def decorate_step_callback(step_callback):
            def __step_callback(t, q, u):
                r = step_callback(t, q, u)
                if self.do_render:
                    self.step_render(t, q, u)
                return r

            return __step_callback

        system.step_callback = decorate_step_callback(system.step_callback)

        self.ren_win.SetShowWindow(False)
        self.add_key_event("q", self.hide)
        super().show(auto_close=False, interactive_update=True)
        self.step_render(system.t0, system.q0, system.u0)

    def step_render(self, t, q, u):
        for twin in self.__visual_twins:
            twin.update_state(t, q[twin.contr.qDOF], u[twin.contr.uDOF])
        self.update()

    def __add_visual_twin(self, visual_twin: _VisualTwinBase):
        if visual_twin not in self.__visual_twins:
            self.__visual_twins.append(visual_twin)
            for actor in visual_twin.actors:
                self.add_actor(actor)
        else:
            raise Exception("visual twin already added!")

    def render_solution(self, solution, repeat=False, play_speed_up=1):
        while True:
            t0_sim = solution.t[0]
            t0_real = perf_counter()
            for ti, qi, ui in zip(solution.t, solution.q, solution.u):
                t_real = perf_counter() - t0_real
                t_sim = ti - t0_sim
                dt = t_real - t_sim / play_speed_up
                if dt > 0.0:
                    continue
                else:
                    sleep(-dt)
                self.step_render(ti, qi, ui)
                if not self.do_render:
                    return
            if not repeat:
                break

    def show(self):
        self.do_render = True
        self.ren_win.SetShowWindow(True)

    def hide(self):
        self.do_render = False
        self.ren_win.SetShowWindow(False)


"""
class RendererBase:
    def __init__(self, system, contributions=None, winsize=(1000, 1000)) -> None:
        self.active = False
        self.system = system
        self.contributions = (
            system.contributions if contributions is None else contributions
        )
        self.__renderer = vtk.vtkRenderer()
        # self.ren.SetBackground(vtkNamedColors().GetColor3d("Grey"))
        self.__renderer.SetBackground(vtk.vtkNamedColors().GetColor3d("DarkGreen"))

        self.fps_actor = vtk.vtkTextActor()
        self.__renderer.AddActor(self.fps_actor)
        self._init_fps()

        self.renwin = vtk.vtkRenderWindow()
        self.renwin.SetWindowName("")
        self.renwin.AddRenderer(self.__renderer)
        self.renwin.MakeRenderWindowInteractor()
        self.renwin.SetSize(*winsize)
        self.interactor = self.renwin.GetInteractor()
        self.observer = self.interactor.AddObserver(
            vtk.vtkCommand.ExitEvent, self.__handle_window_closed
        )
        self.cam_widget = vtk.vtkCameraOrientationWidget()
        self.cam_widget.SetParentRenderer(self.__renderer)
        self.cam_widget.On()
        self.camera = self.__renderer.GetActiveCamera()

        for contr in self.contributions:
            if hasattr(contr, "actors"):
                for actor in contr.actors:
                    self.__renderer.AddActor(actor)

    def _update_fps(self):
        self.n_frame += 1
        self.fps = self.n_frame / (perf_counter() - self.t0)
        self.fps_actor.SetInput(
            f" frame {self.n_frame} / {self.tot_frame}, fps {self.fps:.2f}" + " " * 10
        )

    def _init_fps(self, tot_frame=0):
        self.t0 = perf_counter()
        self.tot_frame = tot_frame
        self.n_frame = 0
        self.fps_actor.SetInput("")

    def __handle_window_closed(self, inter, event):
        self.stop_step_render()
        self.interactor.TerminateApp()

    def render_solution(self, solution, repeat=False):
        self.active = True
        while True:
            self._init_fps(len(solution.t))
            for sol_i in solution:
                self._update_fps()
                self.step_render(sol_i.t, sol_i.q, sol_i.u)
                if not self.active:
                    return
            if not repeat:
                self.active = False
                return

    def step_render(self, t, q, u):
        for contr in self.contributions:
            if hasattr(contr, "step_render"):
                contr.step_render(t, q[contr.qDOF], u[contr.uDOF])
            elif hasattr(contr, "export"):
                points, cells, point_data, cell_data = contr.export(
                    Solution(self.system, t, q, u)
                )
                ugrid = make_ugrid(points, cells, point_data, cell_data)
                if not hasattr(contr, "_vtkfilter"):
                    contr._vtkfilter = vtkGeometryFilter()
                    if isinstance(contr, CosseratRod_PetrovGalerkin):
                        contr._vtkfilter.SetNonlinearSubdivisionLevel(4)
                    mapper = vtkDataSetMapper()
                    actor = vtkActor()
                    actor.SetMapper(mapper)
                    self.__renderer.AddActor(actor)
                    mapper.SetInputConnection(contr._vtkfilter.GetOutputPort())
                contr._vtkfilter.SetInputData(ugrid)
        self.renwin.Render()
        self.interactor.ProcessEvents()

    def start_interaction(self, t, q, u):
        self._init_fps()
        self.step_render(t, q, u)
        self.interactor.Start()


class RendererSync(RendererBase):
    def __init__(self, system, contributions=None, winsize=(1000, 1000)) -> None:
        super().__init__(system, contributions, winsize)

    def start_step_render(self):
        self.active = True

        def decorate_step_callback(system_step_callback):
            def __step_callback(t, q, u):
                if self.active:
                    self.tot_frame += 1
                    self._update_fps()
                    self.step_render(t, q, u)
                return system_step_callback(t, q, u)

            return __step_callback

        self.system.step_callback = decorate_step_callback(self.system.step_callback)
        self._init_fps()

    def stop_step_render(self):
        if self.active:
            self.active = False


class RendererLinux(RendererBase):
    def __init__(self, system, contributions=None, winsize=(1000, 1000)) -> None:
        super().__init__(system, contributions, winsize)
        self.queue = Queue()
        self.exit_event = Event()

        def target(queue, iterrupt):
            self._init_fps()
            while True:
                el = queue.get()
                if el is None:
                    return
                elif iterrupt.is_set():
                    while self.queue.qsize():
                        self.queue.get()
                    return
                self.tot_frame += 1
                self._update_fps()
                self.step_render(*el)

        self.process = Process(target=target, args=(self.queue, self.exit_event))

    def start_step_render(self):
        self.active = True

        def decorate_step_callback(system_step_callback):
            def __step_callback(t, q, u):
                if self.active:
                    self.queue.put((t, q, u))
                return system_step_callback(t, q, u)

            return __step_callback

        self.system.step_callback = decorate_step_callback(self.system.step_callback)

        self.exit_event.clear()
        self.process.start()

    def stop_step_render(self, wait=False):
        if self.active:
            self.active = False
            if self.process.is_alive():
                self.queue.put(None)
                if wait:
                    self.process.join()
                else:
                    self.exit_event.set()

"""

"""
def __decorate_vtk_ugrid(
    contr, ugrid, A_BM=np.eye(3), B_r_CP=np.zeros(3), color=(255, 255, 255)
):
    if not hasattr(contr, "actors"):
        contr.actors = []
    if not hasattr(contr, "H_IB"):
        contr.H_IB = vtk.vtkMatrix4x4()
        contr.H_IB.Identity()
    if not hasattr(contr, "step_render"):

        def step_render(t, q, u):
            A_IB = contr.A_IB(t, q)
            r_OP = contr.r_OP(t, q)[:, None]
            for i in range(3):
                for j in range(3):
                    contr.H_IB.SetElement(i, j, A_IB[i, j])
                contr.H_IB.SetElement(i, 3, r_OP[i])

        contr.step_render = step_render
    H_BM = np.block(
        [
            [A_BM, B_r_CP[:, None]],
            [0, 0, 0, 1],
        ]
    )
    _H_IB = vtk.vtkMatrixToLinearTransform()
    _H_IB.SetInput(contr.H_IB)
    _H_IM = vtk.vtkTransform()
    _H_IM.PostMultiply()
    _H_IM.SetMatrix(H_BM.flatten())
    _H_IM.Concatenate(_H_IB)
    tf_filter = vtk.vtkTransformFilter()
    tf_filter.SetInputData(ugrid)
    tf_filter.SetTransform(_H_IM)

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(tf_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(np.array(color, dtype=np.float64) / 255)
    # base_actor.GetProperty().SetOpacity(0.2)
    contr.actors.append(actor)


def decorate_box(
    contr,
    dimensions=np.ones(3),
    A_BM=np.eye(3),
    B_r_CP=np.zeros(3),
    color=(255, 255, 255),
):
    source = vtk.vtkCubeSource()
    source.SetXLength(dimensions[0])
    source.SetYLength(dimensions[1])
    source.SetZLength(dimensions[2])
    __decorate_vtk_source(contr, source, A_BM, B_r_CP, color)


def decorate_cone(
    contr,
    radius=1,
    height=2,
    resolution=30,
    A_BM=np.eye(3),
    B_r_CP=np.zeros(3),
    color=(255, 255, 255),
):
    source = vtk.vtkConeSource()
    source.SetRadius(radius)
    source.SetHeight(height)
    source.SetResolution(resolution)
    source.SetDirection(0, 0, 1)
    source.SetCenter(0, 0, height / 4)
    __decorate_vtk_source(contr, source, A_BM, B_r_CP, color)


def decorate_cylinder(
    contr,
    radius=1,
    height=2,
    resolution=30,
    A_BM=np.eye(3),
    B_r_CP=np.zeros(3),
    color=(255, 255, 255),
):
    source = vtk.vtkCylinderSource()
    source.SetRadius(radius)
    source.SetHeight(height)
    source.SetResolution(resolution)
    __decorate_vtk_source(contr, source, A_BM, B_r_CP, color)


def decorate_sphere(
    contr,
    radius=1,
    resolution=30,
    A_BM=np.eye(3),
    B_r_CP=np.zeros(3),
    color=(255, 255, 255),
):
    source = vtk.vtkSphereSource()
    source.SetRadius(radius)
    source.SetPhiResolution(int(resolution / 2 - 1))
    source.SetThetaResolution(resolution)
    __decorate_vtk_source(contr, source, A_BM, B_r_CP, color)


def decorate_capsule(
    contr,
    radius=1,
    height=2,
    resolution=30,
    A_BM=np.eye(3),
    B_r_CP=np.zeros(3),
    color=(255, 255, 255),
):
    source = vtk.vtkCylinderSource()
    source.SetRadius(radius)
    source.SetHeight(height)
    source.SetResolution(resolution)
    source.CapsuleCapOn()
    __decorate_vtk_source(contr, source, A_BM, B_r_CP, color)


def decorate_tetrahedron(
    contr, edge=1, A_BM=np.eye(3), B_r_CP=np.zeros(3), color=(255, 255, 255)
):
    # see https://de.wikipedia.org/wiki/Tetraeder
    h_D = edge * np.sqrt(3) / 2
    h_P = edge * np.sqrt(2 / 3)
    r_OM = np.array([0, h_D / 3, h_P / 4])
    p1 = np.array([-edge / 2, 0, 0]) - r_OM
    p2 = np.array([+edge / 2, 0, 0]) - r_OM
    p3 = np.array([0, h_D, 0]) - r_OM
    p4 = np.array([0, h_D / 3, h_P]) - r_OM

    points = vtk.vtkPoints()
    points.InsertNextPoint(*p1)
    points.InsertNextPoint(*p2)
    points.InsertNextPoint(*p3)
    points.InsertNextPoint(*p4)

    # The first tetrahedron
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)

    tetra = vtk.vtkTetra()

    tetra.GetPointIds().SetId(0, 0)
    tetra.GetPointIds().SetId(1, 1)
    tetra.GetPointIds().SetId(2, 2)
    tetra.GetPointIds().SetId(3, 3)

    cellArray = vtk.vtkCellArray()
    cellArray.InsertNextCell(tetra)
    ugrid.SetCells(vtk.VTK_TETRA, cellArray)

    __decorate_vtk_ugrid(contr, ugrid, A_BM, B_r_CP, color)
"""
