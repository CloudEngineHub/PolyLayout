import moderngl
import numpy as np

from ..geometry import Camera, Layout, Pose


class EdgeRenderer:
    def __init__(self, width: int, height: int) -> None:
        from moderngl import _store
        if _store.default_context is None:
            moderngl.create_context(standalone=True)

        self.ctx = moderngl.get_context()
        self.program = self.ctx.program(
            vertex_shader="""
            #version 330
            uniform mat4 model_view_proj;
            in vec3 in_position;
            void main() {
                gl_Position = model_view_proj * vec4(in_position, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            out vec4 f_color;
            uniform vec4 color;
            void main() {
                f_color = color;
            }
            """,
        )
        size = (width, height)
        self.depth_texture = self.ctx.depth_texture(size, alignment=1)
        self.fbo = self.ctx.framebuffer(
            color_attachments=self.ctx.texture(size, components=4, dtype="f4"),
            depth_attachment=self.depth_texture,
        )
        self.output = self.ctx.framebuffer([self.ctx.renderbuffer(size, components=4, dtype="f4")])

    def compute_mvp(self, camera: Camera, pose: Pose) -> np.ndarray:
        assert camera.dist.shape[0] == 0, "Distortion not supported for edge rendering."
        width, height = camera.size
        assert width == self.fbo.size[0] and height == self.fbo.size[1]
        f, c = camera.f, camera.c
        view_to_clip = perspective_projection(f[0], f[1], c[0], c[1], width, height)

        T_w2c = np.eye(4)
        R, t = pose.numpy()
        T_w2c[:3, :3] = R
        T_w2c[:3, 3] = t
        opencv_to_opengl = np.eye(4)
        opencv_to_opengl[1, 1] = opencv_to_opengl[2, 2] = -1
        world_to_view = opencv_to_opengl @ T_w2c

        world_to_clip = view_to_clip @ world_to_view
        return np.ascontiguousarray(world_to_clip.T)

    def render(self, layout: Layout, camera: Camera, pose: Pose, line_width: float) -> np.ndarray:
        """
        Renders the edges of the given layout from the viewpoint of the specified camera and pose.

        Args:
            layout: The room layout to render.
            camera: The camera parameters.
            pose: The camera pose in the world.
            line_width: The width of the rendered edges.
        Returns:
            img: Rendered edge map (H, W).
        """
        verts, edges, faces = (x.numpy() for x in (layout.corners, layout.edges, layout.faces))

        vbo_tri = self.ctx.buffer(verts.astype("f4").tobytes())
        ibo_tri = self.ctx.buffer(faces.astype("i4").tobytes())
        vao_tri = self.ctx.vertex_array(self.program, [(vbo_tri, "3f", "in_position")], ibo_tri)

        vbo_line = self.ctx.buffer(verts.astype("f4").tobytes())
        ibo_line = self.ctx.buffer(edges.astype("i4").tobytes())
        vao_line = self.ctx.vertex_array(self.program, [(vbo_line, "3f", "in_position")], ibo_line)

        mvp = self.compute_mvp(camera, pose)

        self.fbo.use()
        self.ctx.clear()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.line_width = line_width

        self.program["model_view_proj"].write(mvp.astype("f4"))

        # First pass, render faces
        self.ctx.polygon_offset = 10.0, 10.0
        self.program["color"].value = (0.0, 0.0, 0.0, 1.0)
        vao_tri.render(moderngl.TRIANGLES)

        # Second pass, render edges
        self.ctx.polygon_offset = 0, 0
        self.program["color"].value = (1.0, 1.0, 1.0, 1.0)
        vao_line.render(moderngl.LINES)

        self.ctx.copy_framebuffer(self.output, self.fbo)
        data = self.output.read(dtype="f1")
        img = np.frombuffer(data, dtype="uint8").reshape((*self.fbo.size[1::-1], 3))
        img = np.flipud(img)

        vao_tri.release()
        vao_line.release()
        return img


def perspective_projection(
    fx: float, fy: float, cx: float, cy: float, width: int, height: int, znear: float = 0.01, zfar: float = 1000.0
) -> np.ndarray:
    """! Calculate OpenGL perspective matrix from OpenCV/COLMAP camera intrinsics.

    @see http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix

    @param fx, fy Focal lengths.
    @param cx, cy Principal point.
    @param width, height Image dimensions.
    @param znear, zfar Position of the near and far planes.
    @return The projection matrix (4, 4).
    """
    cy = height - cy  # Flip y-axis
    return np.array(
        [
            [2 * fx / width, 0, 1 - 2 * cx / width, 0],
            [0, 2 * fy / height, 1 - 2 * cy / height, 0],
            [0, 0, -(zfar + znear) / (zfar - znear), -2 * zfar * znear / (zfar - znear)],
            [0, 0, -1, 0],
        ]
    )
