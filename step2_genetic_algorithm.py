# ============================================================
# STEP 2: GENETIC ALGORITHM IMPLEMENTATION
# Member 2's responsibility
# ============================================================

import numpy as np
import random

class GeneticAlgorithm:
    """
    Genetic Algorithm for feature selection.

    Each chromosome = binary array of length n_features
    1 = select this feature, 0 = exclude it
    """

    def __init__(self,
                 n_features=13,
                 population_size=30,
                 n_generations=50,
                 crossover_rate=0.8,
                 mutation_rate=0.05,
                 tournament_size=4,
                 elitism=2,
                 random_state=42):

        self.n_features      = n_features
        self.population_size = population_size
        self.n_generations   = n_generations
        self.crossover_rate  = crossover_rate
        self.mutation_rate   = mutation_rate
        self.tournament_size = tournament_size
        self.elitism         = elitism          # top N chromosomes carried unchanged
        random.seed(random_state)
        np.random.seed(random_state)

        # History tracking
        self.best_fitness_history = []
        self.avg_fitness_history  = []
        self.best_chromosome      = None
        self.best_fitness         = 0.0

    # ── Initialization ────────────────────────────────────────
    def initialize_population(self):
        """Create random binary chromosomes. Guarantee at least 1 feature selected."""
        population = []
        for _ in range(self.population_size):
            chrom = np.random.randint(0, 2, self.n_features)
            if chrom.sum() == 0:                 # ensure at least 1 feature
                chrom[np.random.randint(0, self.n_features)] = 1
            population.append(chrom)
        return population

    # ── Selection: Tournament ─────────────────────────────────
    def tournament_selection(self, population, fitnesses):
        """Randomly pick tournament_size individuals; return the fittest."""
        contestants = random.sample(range(len(population)), self.tournament_size)
        winner = max(contestants, key=lambda i: fitnesses[i])
        return population[winner].copy()

    # ── Crossover: Single-point ───────────────────────────────
    def single_point_crossover(self, parent1, parent2):
        """Split parents at a random point and swap tails."""
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.n_features - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        # Ensure children have at least 1 feature
        for child in [child1, child2]:
            if child.sum() == 0:
                child[np.random.randint(0, self.n_features)] = 1
        return child1, child2

    # ── Mutation: Bit-flip ────────────────────────────────────
    def bit_flip_mutation(self, chromosome):
        """Flip each bit with probability mutation_rate."""
        mutated = chromosome.copy()
        for i in range(self.n_features):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        if mutated.sum() == 0:
            mutated[np.random.randint(0, self.n_features)] = 1
        return mutated

    # ── Main Evolution Loop ───────────────────────────────────
    def evolve(self, fitness_function, verbose=True):
        """
        Run the GA.

        Parameters
        ----------
        fitness_function : callable
            Takes a binary chromosome array, returns a float (F1 score).
        verbose : bool
            Print generation-by-generation progress.

        Returns
        -------
        best_chromosome : np.ndarray
        best_fitness    : float
        """
        population = self.initialize_population()

        if verbose:
            print("=" * 55)
            print("GENETIC ALGORITHM — FEATURE SELECTION")
            print("=" * 55)
            print(f"Population size : {self.population_size}")
            print(f"Generations     : {self.n_generations}")
            print(f"Crossover rate  : {self.crossover_rate}")
            print(f"Mutation rate   : {self.mutation_rate}")
            print(f"Elitism         : top {self.elitism} carried over")
            print("=" * 55)

        for gen in range(self.n_generations):

            # Evaluate fitness for every chromosome
            fitnesses = [fitness_function(chrom) for chrom in population]

            # Track best
            gen_best_idx     = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]

            if gen_best_fitness > self.best_fitness:
                self.best_fitness    = gen_best_fitness
                self.best_chromosome = population[gen_best_idx].copy()

            self.best_fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(np.mean(fitnesses))

            if verbose and (gen % 10 == 0 or gen == self.n_generations - 1):
                n_selected = self.best_chromosome.sum()
                print(f"Gen {gen+1:>3} | Best F1: {self.best_fitness:.4f} "
                      f"| Avg F1: {np.mean(fitnesses):.4f} "
                      f"| Features selected: {int(n_selected)}")

            # Build next generation
            # 1. Elitism — keep top chromosomes unchanged
            sorted_idx  = np.argsort(fitnesses)[::-1]
            new_pop     = [population[i].copy() for i in sorted_idx[:self.elitism]]

            # 2. Fill the rest via selection + crossover + mutation
            while len(new_pop) < self.population_size:
                p1 = self.tournament_selection(population, fitnesses)
                p2 = self.tournament_selection(population, fitnesses)
                c1, c2 = self.single_point_crossover(p1, p2)
                new_pop.append(self.bit_flip_mutation(c1))
                if len(new_pop) < self.population_size:
                    new_pop.append(self.bit_flip_mutation(c2))

            population = new_pop

        if verbose:
            print("\n" + "=" * 55)
            print("EVOLUTION COMPLETE")
            print(f"Best F1 Score : {self.best_fitness:.4f}")
            print(f"Features used : {int(self.best_chromosome.sum())} / {self.n_features}")
            print("=" * 55)

        return self.best_chromosome, self.best_fitness


# ── Quick sanity test (runs when script is executed directly) ─
if __name__ == "__main__":
    print("Running GA sanity check with a dummy fitness function...")

    def dummy_fitness(chromosome):
        """Simulate F1: more features = diminishing returns, random noise."""
        return min(0.5 + chromosome.sum() * 0.03 + np.random.uniform(-0.02, 0.02), 1.0)

    ga = GeneticAlgorithm(n_features=13, population_size=20, n_generations=30)
    best_chrom, best_f1 = ga.evolve(dummy_fitness, verbose=True)

    print(f"\nBest chromosome : {best_chrom}")
    print(f"Best F1 (dummy) : {best_f1:.4f}")
    print("\nStep 2 GA module — OK")
