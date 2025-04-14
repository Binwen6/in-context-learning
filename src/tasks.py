import math

import torch


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None,center=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None or center is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "gaussian_kernel_regression": GaussianKernelRegression,
        "nonlinear_dynamics":NonlinearDynamicalSystem,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1,center=None):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
        center=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
        center=None,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
        center=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4,center=None):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class GaussianKernelRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1.0,
        noise_std=0.0,
        center=5,
        bandwidth=1.0,
    ):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.noise_std = noise_std
        self.n_centers = center
        self.bandwidth = bandwidth

        self.centers_b = torch.randn(batch_size, center, n_dims)
        self.weights_b = torch.randn(batch_size, center)

    def evaluate(self, xs_b):
        """
        xs_b: [B, N, D]
        centers_b: [B, C, D]
        weights_b: [B, C]
        """
        B, N, D = xs_b.shape
        _, C, _ = self.centers_b.shape

        xs = xs_b.unsqueeze(2)  # [B, N, 1, D]
        centers = self.centers_b.unsqueeze(1)  # [B, 1, C, D]
        dists_sq = ((xs - centers) ** 2).sum(-1)  # [B, N, C]

        kernels = torch.exp(-dists_sq / (2 * self.bandwidth ** 2))  # [B, N, C]
        ys_b = (kernels * self.weights_b.unsqueeze(1)).sum(-1)  # [B, N]

        if self.noise_std > 0:
            ys_b += torch.randn_like(ys_b) * self.noise_std

        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, n_centers=5, **kwargs):
        return {
            "centers": torch.randn(num_tasks, n_centers, n_dims),
            "weights": torch.randn(num_tasks, n_centers),
        }
        
class NonlinearDynamicalSystem(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1.0,
        noise_std=0.0,
        dynamics_type="poly",
        center=None,
    ):
        super(NonlinearDynamicalSystem, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.noise_std = noise_std
        self.dynamics_type = dynamics_type
        
    def evaluate(self, xs_b):
        if self.dynamics_type == "poly":
            return self.poly(xs_b)
        elif self.dynamics_type == "tanh":
            return self.tanh(xs_b)
        elif self.dynamics_type == "logistic":
            return self.logistic(xs_b)
        elif self.dynamics_type == "duffling":
            return self.duffling(xs_b)
        elif self.dynamics_type == "vdp":
            return self.vdp(xs_b)
        elif self.dynamics_type=="lorenz":
            return self.lorenz(xs_b)
        else:
            raise ValueError(f"Unknown dynamics type: {self.dynamics_type}")
        
    def poly(self, xs_b):
        B, L, D = xs_b.shape
        w1 = torch.randn(B, D, 1, device=xs_b.device)
        w2 = torch.randn(B, D, 1, device=xs_b.device)
        w0 = torch.randn(B, 1, device=xs_b.device)  # shape [B, 1]
        x1 = torch.bmm(xs_b, w1).squeeze(-1)  # [B, L]
        x2 = torch.bmm(xs_b**2, w2).squeeze(-1)
        pred = self.scale * (x1 + x2 + w0)  # broadcast [B, 1] + [B, L]
        return pred
        
    def tanh(self, xs_b):
        B, L, D = xs_b.shape
        w = torch.randn(B, D, 1, device=xs_b.device)
        b = torch.randn(B, 1, device=xs_b.device)
        xw = torch.bmm(xs_b, w).squeeze(-1)  # [B, L]
        return torch.tanh(xw + b)
        
    def logistic(self, xs_b):
        # logistic activation: y = 1 / (1 + exp(-w·x + b))
        B, L, D = xs_b.shape
        w = torch.randn(B, D, 1, device=xs_b.device)
        b = torch.randn(B, 1, device=xs_b.device)
        xw = torch.bmm(xs_b, w).squeeze(-1)  # [B, L]
        return torch.sigmoid(xw + b)  # [B, L]
        
    def duffling(self, xs_b):
        B, L, D = xs_b.shape
        if D < 2:
            raise ValueError("Duffling dynamics requires at least 2 dims")
        x = xs_b[:, :, 0]
        v = xs_b[:, :, 1]
        dx = v
        delta = torch.rand(B, 1, device=xs_b.device) * 0.5 + 0.1
        alpha = torch.rand(B, 1, device=xs_b.device) * 2.0 + 0.5
        beta = torch.rand(B, 1, device=xs_b.device) * 1.0 + 0.2
        dv = -delta * v - alpha * x - beta * x**3
        return dx + dv  # [B, L]
        
    def vdp(self, xs_b):
        B, L, D = xs_b.shape
        if D < 2:
            raise ValueError("VDP dynamics requires at least 2 dims")
        x = xs_b[:, :, 0]
        y = xs_b[:, :, 1]
        dx = y
        mu = torch.rand(B, 1, device=xs_b.device) * 2.0 + 0.5
        dy = mu * (1 - x ** 2) * y - x
        return dx + dy  # [B, L]
        
    def lorenz(self, xs_b):
        B, L, D = xs_b.shape
        if D < 3:
            raise ValueError("Lorenz dynamics requires at least 3 dims")
        x = xs_b[:, :, 0]
        y = xs_b[:, :, 1]
        z = xs_b[:, :, 2]
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx + dy + dz  # [B, L]

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, dynamics_type="poly", **kwargs):
        if dynamics_type == "poly":
            return {
                "w0": torch.randn(num_tasks, n_dims, 1),
                "w1": torch.randn(num_tasks, n_dims, 1),
                "w2": torch.randn(num_tasks, n_dims, 1)  # 修复了torch.random
            }
        elif dynamics_type == "tanh":
            return {
                "w": torch.randn(num_tasks, n_dims, 1),
                "b": torch.randn(num_tasks, n_dims, 1),
            }
        elif dynamics_type == "logistic":
            return {
                "w": torch.randn(num_tasks, n_dims, 1),
                "b": torch.randn(num_tasks, 1, 1),
            }
        elif dynamics_type == "duffling":
            return {
                "delta": torch.rand(num_tasks, 1) * 0.5 + 0.1,   # [0.1, 0.6]
                "alpha": torch.rand(num_tasks, 1) * 2.0 + 0.5,   # [0.5, 2.5]
                "beta":  torch.rand(num_tasks, 1) * 1.0 + 0.2,   # [0.2, 1.2]
            }
        elif dynamics_type == "vdp":
            return {
                "mu": torch.rand(num_tasks, 1),
            }
        else:
            raise ValueError(f"Unknown dynamics type: {dynamics_type}")

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

        
# class NonlinearDynamicalSystem(Task):
#     def __init__(
#         self,
#         n_dims,
#         batch_size,
#         pool_dict=None,
#         seeds=None,
#         scale=1.0,
#         noise_std=0.0,
#         dynamics_type="poly",
#     ):
#         super(NonlinearDynamicalSystem, self).__init__(n_dims, batch_size, pool_dict, seeds)
#         self.scale = scale
#         self.noise_std = noise_std
#         self.dynamics_type = dynamics_type
        
#     def evaluate(self, xs_b):
#         return self.dynamics_type(xs_b)
        
#     def poly(self,xs_b):
#         w1=torch.randn(self.b_size, self.n_dims, 1)
#         w2=torch.randn(self.b_size, self.n_dims, 1)
#         w0=torch.randn(self.b_size, self.n_dims, 1)
#         pred=w0+w1@xs_b+w2@(xs_b**2)
#         return pred
#     def tanh(self,xs_b):
#         w=torch.randn(self.b_size, self.n_dims, 1)
#         b=torch.randn(self.b_size, self.n_dims, 1)
#         pred=torch.tanh(w@xs_b+b)
#         return pred
#     # def exp_sin(self,xs_b):
#     #     # y = e^(-x^2) * sin(x)
#     #     pred = torch.exp(-xs_b**2) * torch.sin(xs_b)
#     #     # pred=pred.sum(dim=-1)
#     #     return pred
#     # def trig(self,xs_b):
#     #     # y = sin(x) + cos(2x)
#     #     pred = torch.sin(xs_b) + torch.cos(2 * xs_b)
#     #     # pred=pred.sum(dim=-1)
#     #     return pred
#     # def piecewise(self,xs_b):
#     #     pred = torch.where(xs_b < 0, xs_b, xs_b**2)
#     #     # pred=pred.sum(dim=-1)
#     #     return pred
#     def logistic(self,xs_b):
#         # logistic activation: y = 1 / (1 + exp(-w·x + b))
#         w = torch.randn(self.b_size, self.n_dims, 1, device=self.device)
#         b = torch.randn(self.b_size, 1, 1, device=self.device)
#         logits = xs_b @ w + b
#         pred = torch.sigmoid(logits)
#         return pred[:, :, 0]
#     def duffling(self,xs_b):
#         x = xs_b[:, :, 0]
#         v = xs_b[:, :, 1]
#         dx = v
#         delta = torch.rand(self.b_size, 1) * 0.5 + 0.1   # [0.1, 0.6]
#         alpha = torch.rand(self.b_size, 1) * 2.0 + 0.5   # [0.5, 2.5]
#         beta  = torch.rand(self.b_size, 1) * 1.0 + 0.2   # [0.2, 1.2]
#         dv = -delta * v - alpha * x - beta * x ** 3
#         return torch.stack([dx, dv], dim=-1)
#     def vdp(self,xs_b):
#         x = xs_b[:, :, 0]s
#         y = xs_b[:, :, 1]
#         dx = y
#         mu=torch.rand(1) 
#         dy = mu * (1 - x ** 2) * y - x
#         return torch.stack([dx, dy], dim=-1)

#     @staticmethod
#     def generate_pool_dict(n_dims, num_tasks, **kwargs):
#         if self.dynamics_type=="poly":
#             return {
#                 w0=torch.randn(num_tasks,n_dims,1)
#                 w1=torch.randn(num_tasks,n_dims,1)
#                 w2=torch.random(num_tasks,n_dims,1)
#             }
#         elif self.dynamics_type=="tanh":
#             return{
#                 w=torch.randn(num_tasks, self.n_dims, 1),
#                 b=torch.randn(self.b_size, self.n_dims, 1),
#             }
#         # elif self.dynamics_type=="exp_sin":
#         #     return{}
#         # elif self.dynamics_type=="trig":
#         #     return{}
#         # elif self.dynamics_type=="piecewise":
#         #     return{}
#         elif self.dynamics_type=="logistic":
#             return{
#                 w = torch.randn(self.b_size, self.n_dims, 1, device=self.device),
#                 b = torch.randn(self.b_size, 1, 1, device=self.device),
#             }
#         elif self.dynamics_type=="duffling":
#             return{
#                 delta = torch.rand(self.b_size, 1) * 0.5 + 0.1,   # [0.1, 0.6]
#         alpha = torch.rand(self.b_size, 1) * 2.0 + 0.5,   # [0.5, 2.5]
#         beta  = torch.rand(self.b_size, 1) * 1.0 + 0.2,   # [0.2, 1.2]
#             }
#         elif self.dynamics_type=="vdp":
#             return{
#                 mu=torch.rand(1),
#             }
#         else:
#             raise ValueError("this function type not allowed.")

#     @staticmethod
#     def get_metric():
#         return squared_error

#     @staticmethod
#     def get_training_metric():
#         return mean_squared_error

