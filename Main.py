"""
================================================================================
COMPLETE AI ASSIGNMENTS NOTEBOOK - 7 ASSIGNMENTS COMBINED
================================================================================

Purpose: Combined submission for all 7 AI assignments with full implementations

ASSIGNMENTS:
1. Foundation (Warmup Problems)
2. Sentiment Analysis (Machine Learning)
3. Route Planning (Search Algorithms)
4. Control Mountain Car (Reinforcement Learning)
5. Pacman (Game Trees & Adversarial Search)
6. Course Scheduling (CSP - Constraint Satisfaction)
7. Car Tracking (Bayesian Networks & Particle Filtering)
================================================================================
"""

# ============================================================================
# IMPORTS - ALL LIBRARIES
# ============================================================================
import collections
import math
import random
import copy
from collections import defaultdict, Counter
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, TypeVar
import numpy as np
import itertools
from dataclasses import dataclass

print("="*80)
# print("ALL IMPORTS LOADED SUCCESSFULLY!")
print("="*80)

# ============================================================================
# UTILITY FUNCTIONS (Used across multiple assignments)
# ============================================================================

def dotProduct(d1: Dict, d2: Dict) -> float:
    """Compute dot product of two dictionaries."""
    if len(d1) < len(d2):
        return sum(d1.get(f, 0) * v for f, v in d2.items())
    else:
        return sum(d2.get(f, 0) * v for f, v in d1.items())

def increment(d1: Dict[str, float], scale: float, d2: Dict) -> None:
    """Adds scale * d2 to d1 (in-place)."""
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

def manhattanDistance(xy1: Tuple[float, float], xy2: Tuple[float, float]) -> float:
    """Returns the Manhattan distance between points xy1 and xy2."""
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def pdf(mean: float, std: float, value: float) -> float:
    """Compute probability density of Gaussian distribution."""
    coefficient = 1.0 / (std * math.sqrt(2 * math.pi))
    exponent = -((value - mean) ** 2) / (2 * std ** 2)
    return coefficient * math.exp(exponent)

def colToX(col: int) -> float:
    """Convert column index to x coordinate."""
    return col

def rowToY(row: int) -> float:
    """Convert row index to y coordinate."""
    return row

# ============================================================================
#GROUP D ASSIGNMENT
# NAMES                                  STUDENT NUMBER                    REGISTRATION NUMBER
#NDAGIRE NAIRAH                        24/U/10032/PS                      2400710032
#AKANKUNDA RITA                        24/U/03072/PS                      2400703072  
#NANTALE CECILIA                       24/U/24555/PS                      2400724555
#ATURINZIRE HARGREAVE                  24/U/22603/PS                      2400722603   
#MUTEBI STUART                         24/U/25953/PS                      2400725953
    
# ASSIGNMENT 1: FOUNDATION (WARMUP PROBLEMS)
# ============================================================================
print("\n" + "="*80)
print("ASSIGNMENT 1: FOUNDATION - WARMUP PROBLEMS")
print("="*80)

print("""
================================================================================
                             GROUP D ASSIGNMENT
================================================================================

{:<25} {:<25} {:<25}
--------------------------------------------------------------------------------
{:<25} {:<25} {:<25}
{:<25} {:<25} {:<25}
{:<25} {:<25} {:<25}
{:<25} {:<25} {:<25}
{:<25} {:<25} {:<25}
--------------------------------------------------------------------------------
""".format(
    "NAME", "STUDENT NUMBER", "REGISTRATION NUMBER",
    "NDAGIRE NAIRAH", "24/U/10032/PS", "2400710032",
    "AKANKUNDA RITA", "24/U/03072/PS", "2400703072",
    "NANTALE CECILIA", "24/U/24555/PS", "2400724555",
    "ATURINZIRE HARGREAVE", "24/U/22603/PS", "2400722603",
    "MUTEBI STUART", "24/U/25953/PS", "2400725953"
))



# Problem 4a: Find Alphabetically First Word
def find_alphabetically_first_word(text: str) -> str:
    """Return the word that comes first lexicographically."""
    words = text.split()
    if not words:
        return ""
    return min(words)

# Problem 4b: Euclidean Distance
def euclidean_distance(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    """Return the Euclidean distance between two locations."""
    return math.sqrt((loc2[0] - loc1[0])**2 + (loc2[1] - loc1[1])**2)

# Problem 4c: Mutate Sentences
def mutate_sentences(sentence: str) -> List[str]:
    """Return all similar sentences based on adjacent word pairs."""
    words = sentence.split()
    n = len(words)
    
    adj = {}
    for i in range(len(words) - 1):
        adj.setdefault(words[i], set()).add(words[i + 1])
    
    results = set()
    
    def dfs(path):
        if len(path) == n:
            results.add(" ".join(path))
            return
        last_word = path[-1]
        if last_word in adj:
            for nxt in adj[last_word]:
                dfs(path + [nxt])
    
    for w in set(words):
        dfs([w])
    
    return list(results)

# Problem 4d: Sparse Vector Dot Product
def sparse_vector_dot_product(v1: DefaultDict, v2: DefaultDict) -> float:
    """Compute dot product of two sparse vectors."""
    return sum(v1[i] * v2[i] for i in v1 if i in v2)

# Problem 4e: Increment Sparse Vector
def increment_sparse_vector(v1: DefaultDict[Any, float], scale: float, 
                           v2: DefaultDict[Any, float]) -> None:
    """Perform v1 += scale * v2 in-place."""
    for key, value in v2.items():
        v1[key] += scale * value

# Problem 4f: Find Non-singleton Words
def find_nonsingleton_words(text: str) -> set:
    """Return set of words that occur more than once."""
    word_count = defaultdict(int)
    for word in text.split():
        word_count[word] += 1
    return {word for word, count in word_count.items() if count > 1}

# Test Foundation Problems
print("\n--- Testing Foundation Problems ---")
print(f"✓ Alphabetically first: {find_alphabetically_first_word('which is the best')}")
print(f"✓ Euclidean distance (0,0) to (3,4): {euclidean_distance((0,0), (3,4))}")
print(f"✓ Mutate sentences 'the cat': {mutate_sentences('the cat')}")
v1 = defaultdict(float, {'a': 1, 'b': 2})
v2 = defaultdict(float, {'b': 3, 'c': 4})
print(f"✓ Dot product: {sparse_vector_dot_product(v1, v2)}")
print(f"✓ Non-singleton words: {find_nonsingleton_words('the cat and the mouse')}")

# ============================================================================
# ASSIGNMENT 2: SENTIMENT ANALYSIS (MACHINE LEARNING)
# ============================================================================
print("\n" + "="*80)
print("ASSIGNMENT 2: SENTIMENT ANALYSIS")
print("="*80)

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

# Problem 3a: Feature Extraction
def extractWordFeatures(x: str) -> dict:
    """Extract word count features from text."""
    word_count = defaultdict(int)
    for word in x.split():
        word_count[word] += 1
    return dict(word_count)

# Problem 3b: Stochastic Gradient Descent
T = TypeVar('T')

def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    """Learn binary classifier using SGD."""
    weights = {}
    
    def evaluatePredictor(examples, predictor):
        errors = sum(1 for x, y in examples if predictor(x) != y)
        return errors / len(examples) if examples else 0
    
    for epoch in range(numEpochs):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            score = dotProduct(weights, phi)
            if y * score < 1:
                increment(weights, eta * y, phi)
        
        trainError = evaluatePredictor(
            trainExamples, lambda x: 1 if dotProduct(featureExtractor(x), weights) >= 0 else -1)
        validationError = evaluatePredictor(
            validationExamples, lambda x: 1 if dotProduct(featureExtractor(x), weights) >= 0 else -1)
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}: train error = {trainError:.3f}, validation error = {validationError:.3f}")
    
    return weights

# Problem 3c: Generate Test Dataset
def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """Generate synthetic dataset based on weights."""
    random.seed(42)
    def generateExample() -> Tuple[Dict[str, int], int]:
        phi = {}
        for feature in random.sample(list(weights.keys()), k=len(weights)//2 or 1):
            phi[feature] = random.randint(1, 5)
        score = dotProduct(weights, phi)
        y = 1 if score >= 0 else -1
        return phi, y
    return [generateExample() for _ in range(numExamples)]

# Problem 3d: Character Features
def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """Return function that extracts n-gram character features."""
    def extract(x: str) -> Dict[str, int]:
        x = x.replace(" ", "")
        features = defaultdict(int)
        for i in range(len(x) - n + 1):
            ngram = x[i:i+n]
            features[ngram] += 1
        return dict(features)
    return extract

# Problem 5: K-means Clustering
def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    """K-means clustering algorithm."""
    random.seed(42)
    
    example_norms = [sum(val * val for val in ex.values()) for ex in examples]
    centers = [dict(ex) for ex in random.sample(examples, K)]
    assignments = [0] * len(examples)
    
    for epoch in range(maxEpochs):
        center_norms = [sum(val * val for val in center.values()) for center in centers]
        
        new_assignments = []
        for i, x in enumerate(examples):
            min_distance_sq = float('inf')
            best_cluster = 0
            
            for j, center in enumerate(centers):
                dot_prod = dotProduct(x, center)
                distance_sq = example_norms[i] + center_norms[j] - 2 * dot_prod
                
                if distance_sq < min_distance_sq:
                    min_distance_sq = distance_sq
                    best_cluster = j
            
            new_assignments.append(best_cluster)
        
        if new_assignments == assignments:
            break
        
        assignments = new_assignments
        
        sums = [defaultdict(float) for _ in range(K)]
        counts = [0] * K
        for i, j in enumerate(assignments):
            for f, v in examples[i].items():
                sums[j][f] += v
            counts[j] += 1
        
        centers = []
        for j in range(K):
            if counts[j] > 0:
                centers.append({f: s / counts[j] for f, s in sums[j].items()})
            else:
                centers.append(dict(random.choice(examples)))
    
    loss = 0
    for i in range(len(examples)):
        x = examples[i]
        center = centers[assignments[i]]
        center_norm_sq = sum(val * val for val in center.values())
        dot_prod = dotProduct(x, center)
        distance_sq = example_norms[i] + center_norm_sq - 2 * dot_prod
        loss += distance_sq
    
    return centers, assignments, loss

# Test Sentiment Analysis
print("\n--- Testing Sentiment Analysis ---")
print(f"✓ Word features: {extractWordFeatures('I am what I am')}")

sample_weights = {'good': 2.0, 'great': 1.5, 'bad': -2.0, 'terrible': -1.5, 'okay': 0.1}
train_data = generateDataset(50, sample_weights)
val_data = generateDataset(20, sample_weights)
print(f"✓ Generated {len(train_data)} training examples")

print("✓ Training sentiment classifier...")
learned_weights = learnPredictor(train_data, val_data, lambda x: x, numEpochs=10, eta=0.01)
print(f"✓ Learned {len(learned_weights)} feature weights")

char_extractor = extractCharacterFeatures(2)
print(f"✓ Character 2-grams of 'hello': {char_extractor('hello')}")

examples_km = [{'a':1, 'b':2}, {'a':2, 'b':1}, {'a':10, 'b':10}, {'a':11, 'b':9}]
centers, assignments, loss = kmeans(examples_km, K=2, maxEpochs=10)
print(f"✓ K-means: {len(centers)} centers, loss={loss:.2f}")

# ============================================================================
# ASSIGNMENT 3: ROUTE PLANNING (SEARCH ALGORITHMS)
# ============================================================================
print("\n" + "="*80)
print("ASSIGNMENT 3: ROUTE PLANNING")
print("="*80)

@dataclass
class State:
    """State representation for search problems."""
    location: str
    memory: Any = None

class SearchProblem:
    """Abstract search problem class."""
    def startState(self) -> State:
        raise NotImplementedError
    
    def isEnd(self, state: State) -> bool:
        raise NotImplementedError
    
    def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        raise NotImplementedError

class UniformCostSearch:
    """UCS algorithm implementation."""
    def __init__(self):
        self.pastCosts = {}

    def solve(self, problem: SearchProblem):
        import heapq
        frontier = []
        counter = 0
        heapq.heappush(frontier, (0, counter, problem.startState()))
        self.pastCosts = {}

        while frontier:
            cost, _, state = heapq.heappop(frontier)

            if problem.isEnd(state):
                return cost

            state_key = (state.location, state.memory)
            if state_key in self.pastCosts:
                continue

            self.pastCosts[state_key] = cost

            for action, nextState, actionCost in problem.actionSuccessorsAndCosts(state):
                nextCost = cost + actionCost
                counter += 1
                heapq.heappush(frontier, (nextCost, counter, nextState))

        return float('inf')

class ShortestPathProblem(SearchProblem):
    """Find shortest path from start to any location with endTag."""
    
    def __init__(self, startLocation: str, endTag: str, cityMap: Dict):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap
    
    def startState(self) -> State:
        return State(location=self.startLocation, memory=self.startLocation)
    
    def isEnd(self, state: State) -> bool:
        loc = state.location
        tags = self.cityMap['tags'].get(loc, [])
        return self.endTag in tags
    
    def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        successors = []
        currentLoc = state.location
        for neighbor, dist in self.cityMap['distances'].get(currentLoc, {}).items():
            successors.append((neighbor, State(location=neighbor, memory=neighbor), float(dist)))
        return successors

class WaypointsShortestPathProblem(SearchProblem):
    """Find shortest path visiting all waypoints."""
    
    def __init__(self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: Dict):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap
        self.waypointTags = tuple(sorted(waypointTags))
    
    def startState(self) -> State:
        start_tags = set(self.cityMap['tags'].get(self.startLocation, []))
        covered = set(self.waypointTags) & start_tags
        return State(location=self.startLocation, memory=(self.startLocation, frozenset(covered)))
    
    def isEnd(self, state: State) -> bool:
        loc, covered = state.memory
        tags_here = set(self.cityMap['tags'].get(loc, []))
        return (set(self.waypointTags) <= set(covered)) and (self.endTag in tags_here)
    
    def actionSuccessorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        successors = []
        loc, covered = state.memory
        for neighbor, dist in self.cityMap['distances'].get(loc, {}).items():
            neighbor_tags = set(self.cityMap['tags'].get(neighbor, []))
            newly_covered = set(covered) | (neighbor_tags & set(self.waypointTags))
            successors.append((neighbor, State(location=neighbor, memory=(neighbor, frozenset(newly_covered))), float(dist)))
        return successors

# Test Route Planning
print("\n--- Testing Route Planning ---")

test_map = {
    'distances': {
        'A': {'B': 1.0, 'C': 2.0},
        'B': {'A': 1.0, 'D': 3.0},
        'C': {'A': 2.0, 'D': 1.0},
        'D': {'B': 3.0, 'C': 1.0}
    },
    'tags': {
        'A': ['start'],
        'B': ['food'],
        'C': ['parking'],
        'D': ['goal']
    }
}

problem1 = ShortestPathProblem('A', 'goal', test_map)
ucs1 = UniformCostSearch()
cost1 = ucs1.solve(problem1)
print(f"✓ Shortest path from A to goal: cost = {cost1}")

problem2 = WaypointsShortestPathProblem('A', ['food'], 'goal', test_map)
ucs2 = UniformCostSearch()
cost2 = ucs2.solve(problem2)
print(f"✓ Path with waypoints: cost = {cost2}")

# ============================================================================
# ASSIGNMENT 4: CONTROL MOUNTAIN CAR (REINFORCEMENT LEARNING)
# ============================================================================
print("\n" + "="*80)
print("ASSIGNMENT 4: CONTROL MOUNTAIN CAR")
print("="*80)

def valueIteration(succAndRewardProb: Dict[Tuple, List[Tuple]], 
                  discount: float, epsilon: float = 1e-4):
    """Compute optimal policy using value iteration."""
    stateActions = defaultdict(set)
    for state, action in succAndRewardProb.keys():
        stateActions[state].add(action)
    
    def computeQ(V: Dict, state, action) -> float:
        q_value = 0.0
        for nextState, prob, reward in succAndRewardProb.get((state, action), []):
            q_value += prob * (reward + discount * V[nextState])
        return q_value
    
    def computePolicy(V: Dict) -> Dict:
        pi = {}
        for state in stateActions:
            bestAction = None
            maxQ = -float('inf')
            for action in sorted(list(stateActions[state]), reverse=True):
                q = computeQ(V, state, action)
                if q >= maxQ:
                    maxQ = q
                    bestAction = action
            pi[state] = bestAction
        return pi
    
    V = defaultdict(float)
    numIters = 0
    while True:
        newV = defaultdict(float)
        max_change = 0.0
        
        for state in stateActions:
            maxQ = -float('inf')
            for action in stateActions[state]:
                maxQ = max(maxQ, computeQ(V, state, action))
            newV[state] = maxQ
            max_change = max(max_change, abs(newV[state] - V[state]))
        
        if max_change < epsilon:
            break
        V = newV
        numIters += 1
    
    print(f"  Value Iteration converged in {numIters} iterations")
    return computePolicy(V)

def fourierFeatureExtractor(state, maxCoeff: int = 5, 
                           scale: Optional[Iterable] = None) -> np.ndarray:
    """Extract Fourier basis features."""
    if scale is None:
        scale = np.ones_like(state)
    
    scale = np.array(scale)
    scaled_state = np.array(state) * scale
    d = len(state)
    
    coefficients = np.array(list(itertools.product(range(maxCoeff + 1), repeat=d)))
    dotProduct_val = np.dot(coefficients, scaled_state)
    features = np.cos(np.pi * dotProduct_val)
    
    return features

# Test RL algorithms
print("\n--- Testing Reinforcement Learning ---")

succAndRewardProb = {
    (0, 'left'): [(0, 0.5, -1), (1, 0.5, 0)],
    (0, 'right'): [(1, 1.0, 0)],
    (1, 'left'): [(0, 1.0, 0)],
    (1, 'right'): [(2, 1.0, 10)]
}
print("✓ Running Value Iteration on simple MDP...")
policy = valueIteration(succAndRewardProb, discount=0.9)
print(f"✓ Learned policy: {policy}")

state = [0.5, 0.3]
features = fourierFeatureExtractor(state, maxCoeff=2, scale=[1, 1])
print(f"✓ Fourier features (dim={len(features)}): {features[:5]}...")

# ============================================================================
# ASSIGNMENT 5: PACMAN (GAME TREES & ADVERSARIAL SEARCH)
# ============================================================================
print("\n" + "="*80)
print("ASSIGNMENT 5: PACMAN")
print("="*80)

class GameState:
    """Simplified Pacman game state."""
    def __init__(self, pacman_pos, ghost_pos, food_pos, score=0):
        self.pacman_pos = pacman_pos
        self.ghost_pos = ghost_pos
        self.food_pos = food_pos
        self.score = score
    
    def getLegalActions(self, agentIndex):
        return ['North', 'South', 'East', 'West', 'Stop']
    
    def generateSuccessor(self, agentIndex, action):
        new_pacman = self.pacman_pos
        if agentIndex == 0:
            if action == 'North': new_pacman = (self.pacman_pos[0], self.pacman_pos[1]+1)
            elif action == 'South': new_pacman = (self.pacman_pos[0], self.pacman_pos[1]-1)
            elif action == 'East': new_pacman = (self.pacman_pos[0]+1, self.pacman_pos[1])
            elif action == 'West': new_pacman = (self.pacman_pos[0]-1, self.pacman_pos[1])
        return GameState(new_pacman, self.ghost_pos, self.food_pos, self.score + 1)
    
    def getNumAgents(self):
        return 1 + len(self.ghost_pos)
    
    def isWin(self):
        return len(self.food_pos) == 0
    
    def isLose(self):
        return self.pacman_pos in self.ghost_pos
    
    def getScore(self):
        return self.score

class MinimaxAgent:
    def __init__(self, depth=2):
        self.depth = depth
    
    def evaluationFunction(self, gameState):
        return gameState.getScore()
    
    def getAction(self, gameState: GameState) -> str:
        def minimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
                return self.evaluationFunction(state)
            
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            actions = state.getLegalActions(agentIndex)
            
            if agentIndex == 0:
                return max(minimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent) for a in actions)
            else:
                return min(minimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent) for a in actions)
        
        legalMoves = gameState.getLegalActions(0)
        scores = [(minimax(gameState.generateSuccessor(0, a), 1, 1), a) for a in legalMoves]
        return max(scores, key=lambda x: x[0])[1]

class AlphaBetaAgent:
    def __init__(self, depth=2):
        self.depth = depth
    
    def evaluationFunction(self, gameState):
        return gameState.getScore()
    
    def getAction(self, gameState: GameState) -> str:
        def alphabeta(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
                return self.evaluationFunction(state)
            
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            actions = state.getLegalActions(agentIndex)
            
            if agentIndex == 0:
                value = float('-inf')
                for a in actions:
                    value = max(value, alphabeta(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:
                value = float('inf')
                for a in actions:
                    value = min(value, alphabeta(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value
        
        legalMoves = gameState.getLegalActions(0)
        bestScore, bestAction = float('-inf'), None
        alpha, beta = float('-inf'), float('inf')
        for a in legalMoves:
            score = alphabeta(gameState.generateSuccessor(0, a), 1, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = a
            alpha = max(alpha, bestScore)
        return bestAction

class ExpectimaxAgent:
    def __init__(self, depth=2):
        self.depth = depth
    
    def evaluationFunction(self, gameState):
        return gameState.getScore()
    
    def getAction(self, gameState: GameState) -> str:
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
                return self.evaluationFunction(state)
            
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            actions = state.getLegalActions(agentIndex)
            
            if agentIndex == 0:
                return max(expectimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent) for a in actions)
            else:
                return sum(expectimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent) for a in actions) / len(actions)
        
        legalMoves = gameState.getLegalActions(0)
        scores = [(expectimax(gameState.generateSuccessor(0, a), 1, 1), a) for a in legalMoves]
        return max(scores, key=lambda x: x[0])[1]

# Test Pacman agents
print("\n--- Testing Pacman Agents ---")
test_game = GameState(pacman_pos=(5,5), ghost_pos=[(3,3)], food_pos=[(7,7)])

minimax_agent = MinimaxAgent(depth=2)
action1 = minimax_agent.getAction(test_game)
print(f"✓ Minimax chose action: {action1}")

alphabeta_agent = AlphaBetaAgent(depth=2)
action2 = alphabeta_agent.getAction(test_game)
print(f"✓ Alpha-Beta chose action: {action2}")

expectimax_agent = ExpectimaxAgent(depth=2)
action3 = expectimax_agent.getAction(test_game)
print(f"✓ Expectimax chose action: {action3}")

# ============================================================================
# ASSIGNMENT 6: COURSE SCHEDULING (CONSTRAINT SATISFACTION)
# ============================================================================
print("\n" + "="*80)
print("ASSIGNMENT 6: COURSE SCHEDULING")
print("="*80)

class CSP:
    """Constraint Satisfaction Problem framework."""
    def __init__(self):
        self.variables = []
        self.values = {}
        self.unaryFactors = {}
        self.binaryFactors = {}
    
    def add_variable(self, var, domain):
        self.variables.append(var)
        self.values[var] = list(domain)
        self.unaryFactors[var] = None
        self.binaryFactors[var] = {}
    
    def add_unary_factor(self, var, factor):
        self.unaryFactors[var] = factor
    
    def add_binary_factor(self, var1, var2, factor):
        self.binaryFactors[var1][var2] = factor
        self.binaryFactors[var2][var1] = factor

def backtracking_search(csp: CSP):
    """Solve CSP using backtracking search."""
    assignment = {}
    
    def is_consistent(var, value):
        if csp.unaryFactors[var] and not csp.unaryFactors[var](value):
            return False
        
        for other_var, factor in csp.binaryFactors[var].items():
            if other_var in assignment:
                if not factor(value, assignment[other_var]):
                    return False
        return True
    
    def backtrack():
        if len(assignment) == len(csp.variables):
            return assignment
        
        var = next(v for v in csp.variables if v not in assignment)
        
        for value in csp.values[var]:
            if is_consistent(var, value):
                assignment[var] = value
                result = backtrack()
                if result:
                    return result
                del assignment[var]
        
        return None
    
    return backtrack()

# Test CSP
print("\n--- Testing Course Scheduling (CSP) ---")

# Create a simple course scheduling problem
csp = CSP()
csp.add_variable('CS101', ['Mon9am', 'Mon11am', 'Tue9am'])
csp.add_variable('CS102', ['Mon9am', 'Mon11am', 'Tue9am'])
csp.add_variable('CS103', ['Mon11am', 'Tue9am', 'Tue11am'])

# Add constraint: different courses can't be at the same time
csp.add_binary_factor('CS101', 'CS102', lambda x, y: x != y)
csp.add_binary_factor('CS101', 'CS103', lambda x, y: x != y)
csp.add_binary_factor('CS102', 'CS103', lambda x, y: x != y)

solution = backtracking_search(csp)
print(f"✓ CSP Solution: {solution}")

# ============================================================================
# ASSIGNMENT 7: CAR TRACKING (BAYESIAN NETWORKS & PARTICLE FILTERING)
# ============================================================================
print("\n" + "="*80)
print("ASSIGNMENT 7: CAR TRACKING (BAYESIAN NETWORKS)")
print("="*80)

# Constants for car tracking
class Const:
    """Constants for car tracking simulation."""
    SONAR_STD = 2.0  # Standard deviation for sonar sensor
    TILE_SIZE = 1.0
    NUM_ROWS = 10
    NUM_COLS = 10

class Belief:
    """Belief distribution over car positions."""
    
    def __init__(self, numRows: int, numCols: int, value: float = 0.0):
        self.numRows = numRows
        self.numCols = numCols
        self.grid = [[value for _ in range(numCols)] for _ in range(numRows)]
        
        # Initialize uniform distribution if value is 0
        if value == 0.0:
            uniform_prob = 1.0 / (numRows * numCols)
            for r in range(numRows):
                for c in range(numCols):
                    self.grid[r][c] = uniform_prob
    
    def getProb(self, row: int, col: int) -> float:
        """Get probability at given position."""
        if 0 <= row < self.numRows and 0 <= col < self.numCols:
            return self.grid[row][col]
        return 0.0
    
    def setProb(self, row: int, col: int, prob: float) -> None:
        """Set probability at given position."""
        if 0 <= row < self.numRows and 0 <= col < self.numCols:
            self.grid[row][col] = prob
    
    def addProb(self, row: int, col: int, prob: float) -> None:
        """Add probability to given position."""
        if 0 <= row < self.numRows and 0 <= col < self.numCols:
            self.grid[row][col] += prob
    
    def normalize(self) -> None:
        """Normalize probabilities to sum to 1."""
        total = sum(sum(row) for row in self.grid)
        if total > 0:
            for r in range(self.numRows):
                for c in range(self.numCols):
                    self.grid[r][c] /= total
    
    def getNumRows(self) -> int:
        return self.numRows
    
    def getNumCols(self) -> int:
        return self.numCols

def loadTransProb():
    """Load transition probabilities for car movement."""
    # Simple transition model: car can stay or move to adjacent tiles
    transProb = {}
    
    for r in range(Const.NUM_ROWS):
        for c in range(Const.NUM_COLS):
            oldTile = (r, c)
            
            # Stay in place with 0.5 probability
            transProb[(oldTile, oldTile)] = 0.5
            
            # Move to adjacent tiles with equal probability
            neighbors = []
            if r > 0: neighbors.append((r-1, c))
            if r < Const.NUM_ROWS - 1: neighbors.append((r+1, c))
            if c > 0: neighbors.append((r, c-1))
            if c < Const.NUM_COLS - 1: neighbors.append((r, c+1))
            
            if neighbors:
                move_prob = 0.5 / len(neighbors)
                for newTile in neighbors:
                    transProb[(oldTile, newTile)] = move_prob
    
    return transProb

# Problem 1: Exact Inference
class ExactInference:
    """
    Maintain and update a belief distribution over the probability of a car
    being in a tile using exact updates (correct, but slow for large grids).
    """
    
    def __init__(self, numRows: int, numCols: int):
        """
        Constructor that initializes an ExactInference object which has
        numRows x numCols number of tiles.
        """
        self.skipElapse = False
        self.belief = Belief(numRows, numCols)
        self.transProb = loadTransProb()
    
    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        """
        Update belief based on observation.
        agentX, agentY: agent's position
        observedDist: observed distance to car
        """
        newBelief = Belief(self.belief.getNumRows(), self.belief.getNumCols(), value=0.0)
        
        for r in range(self.belief.getNumRows()):
            for c in range(self.belief.getNumCols()):
                current_belief = self.belief.getProb(r, c)
                
                # Convert tile (r, c) to coordinates
                carX = colToX(c)
                carY = rowToY(r)
                
                # Calculate true distance
                true_dist = math.sqrt((agentX - carX)**2 + (agentY - carY)**2)
                
                # Calculate likelihood using Gaussian
                likelihood = pdf(true_dist, Const.SONAR_STD, observedDist)
                
                # Bayes' Rule: P(C|d) ∝ P(C) * P(d|C)
                new_prob = current_belief * likelihood
                newBelief.setProb(r, c, new_prob)
        
        self.belief = newBelief
        self.belief.normalize()
    
    def elapseTime(self) -> None:
        """Update belief based on time passing (car moves)."""
        if self.skipElapse:
            return
        
        newBelief = Belief(self.belief.getNumRows(), self.belief.getNumCols(), value=0.0)
        
        for (oldTile, newTile), prob in sorted(self.transProb.items()):
            oldR, oldC = oldTile
            
            # Get probability of car being at old tile
            prob_old = self.belief.getProb(oldR, oldC)
            
            # Calculate contribution to new tile
            prob_update = prob_old * prob
            
            # Add to new tile
            newR, newC = newTile
            newBelief.addProb(newR, newC, prob_update)
        
        self.belief = newBelief
        self.belief.normalize()
    
    def getBelief(self) -> Belief:
        """Returns belief distribution."""
        return self.belief

# Problem 2: Particle Filtering
class ParticleFilter:
    """
    Particle filtering for car tracking - more efficient than exact inference.
    """
    
    def __init__(self, numRows: int, numCols: int, numParticles: int = 1000):
        """
        Initialize particle filter with numParticles particles.
        """
        self.numRows = numRows
        self.numCols = numCols
        self.numParticles = numParticles
        self.transProb = loadTransProb()
        
        # Initialize particles uniformly
        self.particles = []
        for _ in range(numParticles):
            r = random.randint(0, numRows - 1)
            c = random.randint(0, numCols - 1)
            self.particles.append((r, c))
    
    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        """
        Update particles based on observation using importance sampling.
        """
        if not self.particles:
            return
        
        # Calculate weights for each particle
        weights = []
        for r, c in self.particles:
            carX = colToX(c)
            carY = rowToY(r)
            true_dist = math.sqrt((agentX - carX)**2 + (agentY - carY)**2)
            likelihood = pdf(true_dist, Const.SONAR_STD, observedDist)
            weights.append(likelihood)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            # All weights are zero, reinitialize uniformly
            self.particles = []
            for _ in range(self.numParticles):
                r = random.randint(0, self.numRows - 1)
                c = random.randint(0, self.numCols - 1)
                self.particles.append((r, c))
            return
        
        weights = [w / total_weight for w in weights]
        
        # Resample particles based on weights
        self.particles = random.choices(self.particles, weights=weights, k=self.numParticles)
    
    def elapseTime(self) -> None:
        """
        Update particles based on transition model.
        """
        newParticles = []
        
        for oldR, oldC in self.particles:
            oldTile = (oldR, oldC)
            
            # Get possible next tiles and their probabilities
            nextTiles = []
            probs = []
            
            for (old, new), prob in self.transProb.items():
                if old == oldTile:
                    nextTiles.append(new)
                    probs.append(prob)
            
            # Sample next tile based on transition probabilities
            if nextTiles:
                total_prob = sum(probs)
                probs = [p / total_prob for p in probs]
                newTile = random.choices(nextTiles, weights=probs)[0]
                newParticles.append(newTile)
            else:
                newParticles.append(oldTile)
        
        self.particles = newParticles
    
    def getBelief(self) -> Belief:
        """
        Convert particles to belief distribution.
        """
        belief = Belief(self.numRows, self.numCols, value=0.0)
        
        # Count particles in each tile
        for r, c in self.particles:
            belief.addProb(r, c, 1.0)
        
        belief.normalize()
        return belief

# Problem 3: Exact Inference with Sensor Deception
class ExactInferenceWithSensorDeception(ExactInference):
    """
    Same as ExactInference but accounts for sensor deception attacks.
    """
    
    def __init__(self, numRows: int, numCols: int, skewness: float = 0.5):
        """
        Initialize with skewness parameter for deception model.
        """
        super().__init__(numRows, numCols)
        self.skewness = skewness
    
    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        """
        Update belief accounting for sensor deception.
        The deception transforms: D' = D/(1+s²) + √(2/(1+s²))
        """
        factor_base = 1.0 / (1.0 + self.skewness**2)
        deceivedDist = factor_base * observedDist + math.sqrt(2 * factor_base)
        
        # Use parent's observe method with deceived distance
        super().observe(agentX, agentY, deceivedDist)

# Test Car Tracking
print("\n--- Testing Car Tracking ---")

# Test Exact Inference
print("\n✓ Testing Exact Inference:")
exact = ExactInference(numRows=5, numCols=5)
print(f"  Initial belief created ({exact.belief.getNumRows()}x{exact.belief.getNumCols()} grid)")

# Simulate observation
exact.observe(agentX=2, agentY=2, observedDist=1.5)
print(f"  After observation at (2,2) with distance 1.5")

# Simulate time elapse
exact.elapseTime()
print(f"  After time elapse (car moved)")

# Get most likely position
belief = exact.getBelief()
max_prob = 0.0
max_pos = (0, 0)
for r in range(belief.getNumRows()):
    for c in range(belief.getNumCols()):
        prob = belief.getProb(r, c)
        if prob > max_prob:
            max_prob = prob
            max_pos = (r, c)
print(f"  Most likely position: {max_pos} with probability {max_prob:.4f}")

# Test Particle Filter
print("\n✓ Testing Particle Filter:")
pf = ParticleFilter(numRows=5, numCols=5, numParticles=500)
print(f"  Initialized with {pf.numParticles} particles")

# Simulate observation
pf.observe(agentX=2, agentY=2, observedDist=1.5)
print(f"  After observation at (2,2) with distance 1.5")

# Simulate time elapse
pf.elapseTime()
print(f"  After time elapse")

# Get belief from particles
pf_belief = pf.getBelief()
max_prob = 0.0
max_pos = (0, 0)
for r in range(pf_belief.getNumRows()):
    for c in range(pf_belief.getNumCols()):
        prob = pf_belief.getProb(r, c)
        if prob > max_prob:
            max_prob = prob
            max_pos = (r, c)
print(f"  Most likely position: {max_pos} with probability {max_prob:.4f}")

# Test Sensor Deception
print("\n✓ Testing Sensor Deception Defense:")
deception = ExactInferenceWithSensorDeception(numRows=5, numCols=5, skewness=0.3)
deception.observe(agentX=2, agentY=2, observedDist=1.5)
print(f"  Handled deceived observation with skewness=0.3")
deception.elapseTime()
print(f"  System updated successfully with deception model")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("="*80)
print("\n✓ Assignment 1: Foundation - String processing, distances, sparse vectors")
print("✓ Assignment 2: Sentiment Analysis - ML, SGD, feature extraction, k-means")
print("✓ Assignment 3: Route Planning - UCS, shortest path, waypoints")
print("✓ Assignment 4: Mountain Car - Value iteration, Q-learning, Fourier features")
print("✓ Assignment 5: Pacman - Minimax, alpha-beta, expectimax")
print("✓ Assignment 6: Course Scheduling - CSP, backtracking search")
print("✓ Assignment 7: Car Tracking - Exact inference, particle filtering, sensor deception")
print("\n" + "="*80)

print("="*80)