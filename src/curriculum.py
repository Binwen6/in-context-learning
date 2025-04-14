import math


class Curriculum:
    def __init__(self, args,task):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        self.n_dims_truncated = args.dims.start
        self.n_points = args.points.start
        self.n_dims_schedule = args.dims
        self.n_points_schedule = args.points
        self.step_count = 0
        self.task=task
        if task=="nonlinear_dynamics":
            self.dynamics_types = ["poly", "tanh", "logistic", "duffling", "vdp", "lorenz"]
            self.current_dynamics_idx = 0
            self.current_dynamics_type = self.dynamics_types[0]
            self.dynamics_interval=100000

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.n_dims_schedule
        )
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)
        
        if self.task=="nonlinear_dynamics" and self.step_count % self.dynamics_interval == 0:
            self.current_dynamics_idx = min(
                self.current_dynamics_idx + 1, 
                len(self.dynamics_types) - 1
            )
            self.current_dynamics_type = self.dynamics_types[self.current_dynamics_idx]
            

    def update_var(self, var, schedule):
        if self.step_count % schedule.interval == 0:
            var += schedule.inc

        return min(var, schedule.end)
        
    def get_current_dynamics_type(self):
        if self.task=="nonlinear_dynamics":
            return self.current_dynamics_type
        else:
            return None


# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
    final_var = init_var + math.floor((total_steps) / n_steps) * inc

    return min(final_var, lim)
