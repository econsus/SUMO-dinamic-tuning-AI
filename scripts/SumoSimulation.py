import traci


class SUMOSimulation:
    def __init__(self, sumo_config):
        self.sumo_config = sumo_config
        self.simulation_start_time = 0.0
        self.simulation_end_time = 0.0

    def __enter__(self):
        traci.start(self.sumo_config)
        self.simulation_start_time = traci.simulation.getTime()
        print(f"Simulation started at time: {self.simulation_start_time:.2f} seconds")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.simulation_end_time = traci.simulation.getTime()
        simulation_duration = self.simulation_end_time - self.simulation_start_time
        self.on_finish(simulation_duration, exc_type, exc_val)
        traci.close()

    def on_finish(self, duration, exc_type=None, exc_val=None):
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        hours = int(minutes // 60)
        minutes = minutes % 60

        print("\n" + "=" * 60)
        print("SIMULATION FINISHED!")
        print("=" * 60)

        if hours > 0:
            print(f"Total simulation time: {duration:.2f} seconds")
            print(f"   That's {hours} hour(s), {minutes} minute(s), and {seconds} second(s)")
        elif minutes > 0:
            print(f"Total simulation time: {duration:.2f} seconds")
            print(f"   That's {minutes} minute(s) and {seconds} second(s)")
        else:
            print(f"Total simulation time: {duration:.2f} seconds")

        if exc_type is not None:
            print(f"\nSimulation ended with an error: {exc_type.__name__}: {exc_val}")
        else:
            print("\nSimulation completed successfully!")

        print("=" * 60)

    def run(self, simulation_duration, step_callback=None):
        step_length = 0.05
        total_steps = int(simulation_duration / step_length)

        print(f"Running simulation for {simulation_duration} seconds...")
        print(f"   Total steps: {total_steps}")
        print("-" * 60)

        for step in range(total_steps):
            traci.simulationStep()

            if step_callback:
                step_callback()

            if step % 1000 == 0 and step > 0:
                current_time = traci.simulation.getTime()
                progress = (current_time / simulation_duration) * 100
                print(f"Progress: {current_time:.2f}/{simulation_duration} seconds ({progress:.1f}%)")