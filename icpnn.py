from dataclasses import dataclass
import random
import timeit
import numpy as np
import shapely
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import cv2

@dataclass
class Tensor:
    value: np.ndarray
    lr: float
    grad_max: float = +np.Infinity
    grad_min: float = -np.Infinity

    def __post_init__(self):
        self.grad_acc = 0
        self._grad = np.zeros_like(self.value[0]) 

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, grad):
        self.grad_acc += grad
        if (self.grad_acc > self.grad_max) or (self.grad_acc < self.grad_min):
            self.grad_acc -= grad
            return
        self._grad += grad

    def __call__(self):
        return self.value

    def zero_grad(self):

        self.grad.fill(0)

@dataclass
class AffineTransform:
    polygon: np.ndarray
    
    def __post_init__(self):
        self.transformation_matrix = np.eye(N=3)

    def translate(self, dx, dy):
        translation_matrix = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0,  1]
        ])
        self.transformation_matrix @= translation_matrix
    
    def _rotate(self, dr):
        cos = np.cos(dr)
        sin = np.sin(dr)
        if abs(cos) < 2.5e-16:
            cos = 0.0
        if abs(sin) < 2.5e-16:
            sin = 0.0
        x0, y0 = np.mean(self.polygon, axis=0)
        rotation_matrix = np.array([
            [cos, -sin, (x0 - x0 * cos + y0 * sin)],
            [sin,  cos, (y0 - x0 * sin - y0 * cos)],
            [0,    0,   1                         ]
        ])
        self.transformation_matrix @= rotation_matrix

    def rotate_x(self, drx):
        """
        Rotaciona o objeto ao redor do eixo X a partir do centroide.
        :param drx: Ângulo de rotação em radianos ao redor do eixo X.
        """
        cos = np.cos(drx)
        sin = np.sin(drx)
        if abs(cos) < 2.5e-16:
            cos = 0.0
        if abs(sin) < 2.5e-16:
            sin = 0.0
        x0, y0 = np.mean(self.polygon, axis=0)
        origin = np.array([
            [1, 0, -x0],
            [0, 1, -y0],
            [0, 0,   1]
        ])
        rotation_matrix = np.array([
            [1,    0,    0],
            [0,    cos, -sin],
            [0,    sin,  cos]
        ])
        go_back = np.array([
            [1, 0, x0],
            [0, 1, y0],
            [0, 0,  1]
        ])
        self.transformation_matrix @= go_back @ rotation_matrix @ origin
    
    def rotate_y(self, dry):
        """
        Rotaciona o objeto ao redor do eixo Y a partir do centroide.
        :param dry: Ângulo de rotação em radianos ao redor do eixo Y.
        """
        cos = np.cos(dry)
        sin = np.sin(dry)
        if abs(cos) < 2.5e-16:
            cos = 0.0
        if abs(sin) < 2.5e-16:
            sin = 0.0
        x0, y0 = np.mean(self.polygon, axis=0)
        origin = np.array([
            [1, 0, -x0],
            [0, 1, -y0],
            [0, 0,   1]
        ])
        rotation_matrix = np.array([
            [cos, 0, sin],
            [0,   1, 0],
            [-sin, 0, cos]
        ])
        go_back = np.array([
            [1, 0, x0],
            [0, 1, y0],
            [0, 0,  1]
        ])
        # Atualiza a matriz de transformação com a rotação ao redor do eixo Y
        self.transformation_matrix @= go_back @ rotation_matrix @ origin

    def rotate_z(self, drz):
        """
        Rotaciona o objeto ao redor do eixo Z a partir do centroide.
        :param drz: Ângulo de rotação em radianos ao redor do eixo Z.
        """
        cos = np.cos(drz)
        sin = np.sin(drz)
        if abs(cos) < 2.5e-16:
            cos = 0.0
        if abs(sin) < 2.5e-16:
            sin = 0.0
        x0, y0 = np.mean(self.polygon, axis=0)
        origin = np.array([
            [1, 0, -x0],
            [0, 1, -y0],
            [0, 0,   1]
        ])
        rotation_matrix = np.array([
            [cos, -sin, 0],
            [sin,  cos, 0],
            [0,    0,   1]
        ])
        go_back = np.array([
            [1, 0, x0],
            [0, 1, y0],
            [0, 0,  1]
        ])
        # Atualiza a matriz de transformação com a rotação ao redor do eixo Z
        self.transformation_matrix @= go_back @ rotation_matrix @ origin

    def _rotate_z(self, drz):
        """
        Rotaciona o objeto ao redor do eixo Z a partir do centroide.
        :param drz: Ângulo de rotação em radianos ao redor do eixo Z.
        """
        cos = np.cos(drz)
        sin = np.sin(drz)
        if abs(cos) < 2.5e-16:
            cos = 0.0
        if abs(sin) < 2.5e-16:
            sin = 0.0
        x0, y0 = np.mean(self.polygon, axis=0)
        rotation_matrix = np.array([
            [cos, -sin, (x0 - x0 * cos + y0 * sin)],
            [sin,  cos, (y0 - x0 * sin - y0 * cos)],
            [0,    0,   1]
        ])
        
        # Atualiza a matriz de transformação com a rotação ao redor do eixo Z
        self.transformation_matrix @= rotation_matrix

    def scale(self, ds):
        x0, y0 = np.mean(self.polygon, axis=0)
        scale_matrix = np.array([
            [ds, 0,  (x0 - x0 * ds)],
            [0,  ds, (y0 - y0 * ds)],
            [0,  0,  1             ]
        ])
        self.transformation_matrix @= scale_matrix

@dataclass
class IcpNN:
    source: np.ndarray
    target: np.ndarray
    tolerance: float = 10e-5
    h: float = 10e-4
    lr_t: float = 0.01
    lr_rx: float = 0.01
    lr_ry: float = 0.01
    lr_rz: float = 0.2
    lr_s: float = 0.1
    lr_a: float = 0.1

    def __post_init__(self):
        self.affinity = AffineTransform(polygon=self.source)
        # self.preprocess_centroids_alignment()
        self.dt = Tensor(value=self.source, lr=self.lr_t)
        # self.dr = Tensor(value=np.zeros(self.source.shape[0]), lr=self.lr_r)
        self.drx = Tensor(value=np.zeros(self.source.shape[0]), lr=self.lr_rx, grad_max=1, grad_min=-1)
        self.dry = Tensor(value=np.zeros(self.source.shape[0]), lr=self.lr_ry, grad_max=1, grad_min=-1)
        self.drz = Tensor(value=np.zeros(self.source.shape[0]), lr=self.lr_rz)
        
        self.ds = Tensor(value=np.zeros(self.source.shape[0]), lr=self.lr_s)
        self.da = Tensor(value=np.zeros(self.source.shape[0]), lr=self.lr_a)

    def preprocess_centroids_alignment(self):
        source_centroid = np.mean(self.source, axis=0)
        target_centroid = np.mean(self.target, axis=0)
        dx, dy = target_centroid - source_centroid
        self.affinity.translate(dx=dx, dy=dy)

    def step_translation(self):
        dx, dy = - self.dt.grad * self.dt.lr
        self.affinity.translate(dx=dx, dy=dy)
 
    def step_rotation_x(self):
        dr = - self.drx.grad * self.drx.lr
        self.affinity.rotate_x(drx=dr)

    def step_rotation_y(self):
        dr = - self.dry.grad * self.dry.lr
        self.affinity.rotate_y(dry=dr)

    def step_rotation_z(self):
        dr = - self.drz.grad * self.drz.lr
        self.affinity.rotate_z(drz=dr)

    def step_scale(self):
        ds = - self.ds.grad * self.ds.lr + 1
        if ds > 0:
            self.affinity.scale(ds=ds)

    def step_remove(self):
        ...

    def backward_translation(self, loss):
        # dx
        self.affinity.translate(dx=+self.h, dy=0)
        loss_h = self.loss()
        self.affinity.translate(dx=-self.h, dy=0)
        self.dt.grad[0] = (loss_h - loss) / self.h
        # dy
        self.affinity.translate(dx=0, dy=+self.h)
        loss_h = self.loss()
        self.affinity.translate(dx=0, dy=-self.h)
        self.dt.grad[1] = (loss_h - loss) / self.h

    def backward_rotation_x(self, loss):
        self.affinity.rotate_x(drx=+self.h)
        loss_h = self.loss()
        self.affinity.rotate_x(drx=-self.h)
        self.drx.grad = (loss_h - loss) / self.h

    def backward_rotation_y(self, loss):
        self.affinity.rotate_y(dry=+self.h)
        loss_h = self.loss()
        self.affinity.rotate_y(dry=-self.h)
        self.dry.grad = (loss_h - loss) / self.h

    def backward_rotation_z(self, loss):
        self.affinity.rotate_z(drz=+self.h)
        loss_h = self.loss()
        self.affinity.rotate_z(drz=-self.h)
        self.drz.grad = (loss_h - loss) / self.h

    def backward_scale(self, loss):
        self.affinity.scale(ds=+self.h+1)
        loss_h = self.loss()
        self.affinity.scale(ds=1/(+self.h+1))
        self.ds.grad = (loss_h - loss) / self.h

    def step(self):
        self.step_translation()
        self.step_rotation_x()
        self.step_rotation_y()
        self.step_rotation_z()
        self.step_scale()

    def backward(self):
        loss = self.loss()
        self.backward_translation(loss)
        self.backward_rotation_x(loss)
        self.backward_rotation_y(loss)
        self.backward_rotation_z(loss)
        self.backward_scale(loss)

    def forward(self):
        ones_stacked = np.hstack([self.source, np.ones((self.source.shape[0], 1))])
        source_transformed = ones_stacked @ self.affinity.transformation_matrix.T
        return source_transformed[:, 0:2]
    
    def loss(self):
        def polygon_area(p):
            x = p[:, 0]
            y = p[:, 1]
            correction = x[-1] * y[0] - y[-1]* x[0]
            main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
            return 0.5 * np.abs(main_area + correction)
        
        polygon = self.forward()
        
        l1 = np.abs(polygon_area(polygon) - polygon_area(self.target))
        #l1 = np.sum(np.abs(polygon - self.target))/10
        l2 = np.sum((polygon - self.target) ** 2)
        return l1 + l2
    
    def zero_grad(self):
        self.dt.zero_grad()
        self.drx.zero_grad()
        self.dry.zero_grad()
        self.drz.zero_grad()
        self.ds.zero_grad()
        self.da.zero_grad()
    
    def train(self, epochs):
        plot_results_displacement(self.source, self.target, label='epoch: 0', waitkey=0)
        for i in range(epochs):
            self.zero_grad()
            self.backward()
            self.step()
            print(f"loss: {self.loss()}")
            if i%1==0:
                plot_results_displacement(
                    self.forward(), 
                    self.target, 
                    label=f"epoch: {i}",
                    waitkey=1
                )
            if self.loss() <= self.tolerance:
                break
        return self

def plot_results_displacement(p1, p2, label, waitkey):
    fig, axes = plt.subplots(1, 1, figsize=(9, 9))
    axes.add_patch(plt_patches.Polygon(p2))
    axes.add_patch(plt_patches.Polygon(p1, color=(0,1,0, 0.5)))
    axes.add_patch(plt_patches.Rectangle((0, 0), 1, 1, edgecolor=(0,0,1), fill=False))
    plt_patches.Patch
    axes.set_xlim(-0.2, 1.2 )
    axes.set_ylim(-0.2, 1.2 )
    axes.set_title(label)
    plt.gca().set_aspect('equal')
    fig.canvas.draw()
    img_plot = np.array(fig.canvas.renderer.buffer_rgba()) # type: ignore
    cv2.imshow('image', img_plot)
    cv2.waitKey(waitkey)
    plt.close(fig)



if __name__ == '__main__':
    losses = []
    times = []
    tt0 = timeit.default_timer()
    for _ in range(1000):
        # p1 = np.array([
        #     (0.032967034727334976, 0.9882260859012605),
        #     (0.5290423929691315, 0.9175824522972107),
        #     (0.6985871493816376, 0.8485086560249329),
        #     (0.3854003250598908, 0.8453689217567445),
        #     (0.40737834572792053, 0.7731554508209229),
        #     (0.4105180650949478, 0.5855573415756227),
        #     (0.60753533244133, 0.5863422453403473),
        #     (0.7346938848495482, 0.5863422453403472),
        #     (0.43249608576297766, 0.5125588774681091), 
        # ])
        # p2 = np.array([
        #     (0.19020866602659225, 0.5425361096858978),        
        #     (0.6950240456744243, 0.49438203008551473),        
        #     (0.8892455527595423, 0.4301765628305128),
        #     (0.5529694855213165, 0.41492778062820435),        
        #     (0.5810593664646149, 0.3410914838314057),
        #     (0.5868711032141682, 0.1496651342025693),
        #     (0.7953450977802277, 0.15650080144405365),        
        #     (0.9237560033798219, 0.16051363945007327),        
        #     (0.6147672533988953, 0.07383627817034723),
        # ])
        p1 = np.array([
            [0.47784624, 0.97052932],
            [0.50134048, 0.90348525],
            [0.50448733, 0.75671755],
            [0.40815306, 0.60993368],
            [0.22788204, 0.51876676],
            [0.07104558, 0.32037534],
            [0.14075067, 0.08579088],
        ])
        p2 = np.array([
            [0.43246187, 0.76797386],
            [0.45206972, 0.71459695],
            [0.45098039, 0.60239651],
            [0.36601307, 0.48148148],
            [0.23965142, 0.42156863],
            [0.11437908, 0.26688453],
            [0.16666667, 0.10457516],
        ])
        random_seed = np.random.randint(0, 1000)
        np.random.seed(random_seed)
        np.random.shuffle(p1)
        np.random.seed(random_seed)
        np.random.shuffle(p2)

        p2 = shapely.Polygon(p2)
        p2 = shapely.affinity.translate(p2, random.uniform(0, 0.5), random.uniform(0, 0.5))
        p2 = shapely.affinity.rotate(p2, angle=random.randint(0, 360))
        rscale = random.uniform(0.5, 2)
        p2 = shapely.affinity.scale(p2, rscale, rscale)
        p2 = np.array(p2.exterior.coords[:-1])
        
        rrr = AffineTransform(p2)
        rrr.rotate_x(drx=0.5)

        ones_stacked = np.hstack([p2, np.ones((p2.shape[0], 1))])
        source_transformed = ones_stacked @ rrr.transformation_matrix.T
        p2 =  source_transformed[:, 0:2]
    

        t0 = timeit.default_timer()
        nn = IcpNN(source=p1, target=p2)
        nn.train(epochs=100)
        t1 = timeit.default_timer()

        losses.append(np.array(nn.loss()))
        times.append(t1-t0)
        
    tt1 = timeit.default_timer()
    losses = np.array(losses)
    print(f">1: {len(losses[losses > 1])}/{len(losses)}")
    print(f">0.1: {len(losses[losses > 0.1])}/{len(losses)}")
    print(f"<0.1: {len(losses[losses <= 0.1])}/{len(losses)}")
    print(f"<0.01: {len(losses[losses <= 0.01])}/{len(losses)}")
    print(f"<0.001: {len(losses[losses <= 0.001])}/{len(losses)}")
    print(f"mean: {np.mean(losses)}")
    print(f"tM: {np.mean(times)}")
    print(f"tTotal: {tt1-tt0:.2f}")
