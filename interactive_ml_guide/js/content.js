export const modules = {
    module1: {
        title: "1. Core Models",
        content: `
            <div class="prose max-w-none">
                <h2 class="text-3xl font-extrabold text-indigo-900 mb-6">Core Machine Learning Models</h2>
                <p class="text-lg text-gray-700 mb-8">
                    To truly understand Machine Learning, we must peel back the layers of abstraction. This module explores the foundational mathematical frameworks and intuitive mechanics behind core predictive models. We will dissect Linear Regression, Random Forests, XGBoost, and Neural Networks, exploring the fundamental trade-off between bias and variance.
                </p>

                <div class="bg-yellow-50 border-l-4 border-yellow-500 p-6 mb-8 rounded shadow-sm">
                    <h3 class="text-xl font-bold text-yellow-800 mt-0">The "Masala Dosa" Culinary Analogy</h3>
                    <p class="text-gray-700">Throughout this module, we will ground complex mathematics in a familiar context: predicting the perfect cooking time for a Masala Dosa based on pan temperature and batter consistency.</p>
                </div>

                <h3>1. Linear Regression: The Foundation of Continuous Prediction</h3>
                <p>
                    Linear Regression is the bedrock of predictive modeling. It assumes a linear relationship between input features (independent variables, \\( X \\)) and a continuous output (dependent variable, \\( y \\)). 
                </p>
                <div class="math-block text-center text-lg">
                    \\[ y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_n x_n + \\epsilon \\]
                </div>
                <p>
                    Here, \\( \\beta_0 \\) is the intercept (bias), and \\( \\beta_n \\) are the learned weights. The error term \\( \\epsilon \\) represents unobserved variance. The goal is to find the optimal weights that minimize the <strong>Mean Squared Error (MSE)</strong> between predictions \\( \\hat{y} \\) and actuals \\( y \\).
                </p>
                <p>
                    <strong>Intuition (Masala Dosa):</strong> Imagine plotting Pan Temperature on the X-axis and Cooking Time on the Y-axis. Linear regression attempts to draw the "best fit" straight line through these points. It assumes that as the pan gets hotter, the cooking time decreases at a constant rate.
                </p>

                <div class="interactive-panel">
                    <h4 class="font-bold text-lg mb-4">Interactive Linear Regression</h4>
                    <p class="mb-4 text-sm text-gray-600">Click on the canvas to add data points (Temperature vs. Time). Watch as the model continuously updates the weights to minimize the Mean Squared Error.</p>
                    <div class="canvas-container h-80 bg-gray-50 border rounded relative">
                        <canvas id="regression-canvas"></canvas>
                        <div class="absolute top-4 left-4 bg-white/90 p-2 text-sm rounded border shadow">
                            MSE: <span id="reg-mse" class="font-mono">0.00</span><br>
                            y = <span id="reg-m" class="font-mono">0.00</span>x + <span id="reg-b" class="font-mono">0.00</span>
                        </div>
                    </div>
                    <div class="mt-4 flex space-x-4">
                        <button id="btn-reg-clear" class="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition">Clear Points</button>
                    </div>
                </div>

                <h3>2. Ensemble Methods: Bagging and Random Forests</h3>
                <p>
                    A single decision tree is prone to high variance—it easily memorizes the training data (overfitting). <strong>Random Forests</strong> solve this through <em>Bootstrap Aggregating (Bagging)</em>.
                </p>
                <p>
                    By training multiple independent decision trees on random subsets of the data (and using a random subset of features for each split), the ensemble produces predictions that are the average of its constituent trees. This significantly reduces variance without substantially increasing bias. Splits are determined by maximizing <strong>Information Gain</strong> or minimizing <strong>Gini Impurity</strong>:
                </p>
                <div class="math-block text-center text-lg">
                    \\[ Gini = 1 - \\sum_{i=1}^{C} (p_i)^2 \\]
                </div>
                <p>
                    <strong>Intuition (Masala Dosa):</strong> Instead of asking one chef to determine the cooking time (who might have idiosyncratic preferences), you ask 100 chefs. Each chef only looks at a few aspects—one checks temperature, another checks batter viscosity. You average their answers. The collective wisdom is highly robust.
                </p>

                <h3>3. Gradient Boosting & XGBoost: Sequential Optimization</h3>
                <p>
                    While Bagging builds trees independently, <strong>Boosting</strong> builds them sequentially. eXtreme Gradient Boosting (XGBoost) creates weak learners (shallow trees) where each subsequent tree attempts to predict the <em>residuals (errors)</em> of the sum of the previous trees.
                </p>
                <p>
                    This is effectively performing Gradient Descent in functional space. The objective function contains a training loss and a regularization term \\( \\Omega \\) to penalize complexity:
                </p>
                <div class="math-block text-center text-lg">
                    \\[ \\mathcal{L}^{(t)} = \\sum_{i=1}^{n} l(y_i, \\hat{y}_i^{(t-1)} + f_t(x_i)) + \\Omega(f_t) \\]
                </div>
                <p>
                    <strong>Intuition (Masala Dosa):</strong> Chef 1 makes a dosa. It's undercooked by 30 seconds. Chef 2 explicitly focuses <em>only</em> on how to fix that 30-second error. Chef 3 fixes the minor mistakes of Chef 2. This sequential error-correction results in extraordinary precision.
                </p>

                <h3>4. Neural Networks: Perceptrons & Universal Approximation</h3>
                <p>
                    Artificial Neural Networks (ANNs) are built from layers of artificial neurons (perceptrons). Each neuron computes a weighted sum of its inputs, adds a bias, and passes the result through a non-linear activation function (like ReLU or Sigmoid):
                </p>
                <div class="math-block text-center text-lg">
                    \\[ a^{(l)} = \\sigma(W^{(l)}a^{(l-1)} + b^{(l)}) \\]
                </div>
                <p>
                    The <strong>Universal Approximation Theorem</strong> states that a feed-forward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of \\(\\mathbb{R}^n\\). The depth (number of layers) allows the network to learn compositional feature representations.
                </p>
                <p>
                    <strong>Intuition (Masala Dosa):</strong> A highly complex factory pipeline. The first layer detects basic properties (temperature, batter density). The second layer combines these to detect abstract properties (crispiness potential). The final layer outputs the exact cooking time.
                </p>
                
                <div class="interactive-panel">
                    <h4 class="font-bold text-lg mb-4">Neural Network Data Flow Simulation</h4>
                    <p class="mb-4 text-sm text-gray-600">Watch how data propagates through a simple Multi-Layer Perceptron (MLP). The thickness of lines represents weights, and glowing nodes represent activation levels.</p>
                    <div class="canvas-container h-64 bg-slate-900 border rounded border-gray-700">
                        <canvas id="nn-canvas"></canvas>
                    </div>
                </div>

                <div class="bg-blue-50 p-6 rounded-lg mt-8 border border-blue-200">
                    <h4 class="text-xl font-bold text-blue-900 mb-2">The Bias-Variance Tradeoff</h4>
                    <p class="text-blue-800">
                        <strong>Bias:</strong> Error from erroneous assumptions (underfitting). e.g., Using Linear Regression for a highly non-linear pattern.<br>
                        <strong>Variance:</strong> Error from sensitivity to small fluctuations in the training set (overfitting). e.g., A Deep Decision Tree memorizing noise.<br>
                        <em>Total Error = Bias² + Variance + Irreducible Error.</em> The holy grail of ML is finding the sweet spot balancing the two.
                    </p>
                </div>
            </div>
        `
    },
    module2: {
        title: "2. Paradigms & Classification",
        content: `
            <div class="prose max-w-none">
                <h2 class="text-3xl font-extrabold text-indigo-900 mb-6">Learning Paradigms & Classification</h2>
                <p class="text-lg text-gray-700 mb-8">
                    Machine Learning is not a monolithic discipline. The way an algorithm learns depends entirely on the nature of the data provided. In this module, we dissect the four primary learning paradigms and take a rigorous look at Classification dynamics versus Regression.
                </p>

                <h3>1. The Four Pillars of Learning</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 my-8">
                    <div class="bg-white p-6 border rounded shadow-sm hover:shadow-md transition">
                        <h4 class="text-xl font-bold text-indigo-700">Supervised Learning</h4>
                        <p><strong>Mechanism:</strong> Learning from labeled data. We provide input \\(X\\) and target output \\(y\\). The model learns a mapping function \\( f: X \\rightarrow y \\).</p>
                        <p class="mt-2 text-sm text-gray-600"><strong>Use cases:</strong> Image classification, price prediction, spam detection.</p>
                    </div>
                    <div class="bg-white p-6 border rounded shadow-sm hover:shadow-md transition">
                        <h4 class="text-xl font-bold text-emerald-700">Unsupervised Learning</h4>
                        <p><strong>Mechanism:</strong> Finding hidden structures in unlabeled data. Only input \\(X\\) is provided. Algorithms discover clusters, latent variables, or reduced dimensions.</p>
                        <p class="mt-2 text-sm text-gray-600"><strong>Use cases:</strong> Customer segmentation, anomaly detection, dimensionality reduction (PCA).</p>
                    </div>
                    <div class="bg-white p-6 border rounded shadow-sm hover:shadow-md transition">
                        <h4 class="text-xl font-bold text-amber-700">Semi-Supervised Learning</h4>
                        <p><strong>Mechanism:</strong> A small amount of labeled data is combined with a large amount of unlabeled data. Pseudo-labeling is often used to propagate labels.</p>
                        <p class="mt-2 text-sm text-gray-600"><strong>Use cases:</strong> Medical image analysis (where expert labels are expensive but images are abundant).</p>
                    </div>
                    <div class="bg-white p-6 border rounded shadow-sm hover:shadow-md transition">
                        <h4 class="text-xl font-bold text-rose-700">Reinforcement Learning</h4>
                        <p><strong>Mechanism:</strong> An agent interacts with an environment, taking actions to maximize cumulative reward. There is no predefined correct answer, only rewards/penalties over time.</p>
                        <p class="mt-2 text-sm text-gray-600"><strong>Use cases:</strong> Robotics, game AI (AlphaGo), autonomous driving.</p>
                    </div>
                </div>

                <h3>2. Regression vs. Classification: The Mathematical Divide</h3>
                <p>
                    While regression predicts a continuous quantity, <strong>Classification</strong> predicts discrete class labels. This fundamentally changes the mathematical objective of the models.
                </p>
                <div class="overflow-x-auto my-6 rounded shadow-sm border border-gray-200">
                    <table class="min-w-full bg-white">
                        <thead class="bg-slate-100 border-b">
                            <tr>
                                <th class="py-3 px-4 text-left font-semibold text-gray-700">Property</th>
                                <th class="py-3 px-4 text-left font-semibold text-gray-700">Regression</th>
                                <th class="py-3 px-4 text-left font-semibold text-gray-700">Classification</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-200">
                            <tr>
                                <td class="py-3 px-4 font-medium">Output</td>
                                <td class="py-3 px-4">Continuous Real Number \\( \\mathbb{R} \\)</td>
                                <td class="py-3 px-4">Discrete Probabilities / Labels \\( \\{0, 1, \\dots, K\\} \\)</td>
                            </tr>
                            <tr>
                                <td class="py-3 px-4 font-medium">Standard Loss Function</td>
                                <td class="py-3 px-4">Mean Squared Error (MSE)</td>
                                <td class="py-3 px-4">Binary / Categorical Cross-Entropy</td>
                            </tr>
                            <tr>
                                <td class="py-3 px-4 font-medium">Last Layer Activation</td>
                                <td class="py-3 px-4">Linear / None</td>
                                <td class="py-3 px-4">Sigmoid (Binary) / Softmax (Multi-class)</td>
                            </tr>
                            <tr>
                                <td class="py-3 px-4 font-medium">Evaluation Metrics</td>
                                <td class="py-3 px-4">RMSE, MAE, R²</td>
                                <td class="py-3 px-4">Accuracy, Precision, Recall, F1-Score, AUC-ROC</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <h4>Decision Boundaries & The Cross-Entropy Loss</h4>
                <p>
                    In classification, we establish a <strong>Decision Boundary</strong>—a hypersurface that partitions the underlying vector space into sets, one for each class. In Logistic Regression, this boundary is linear.
                </p>
                <p>
                    To train classifiers, we cannot use MSE effectively because the loss landscape becomes non-convex (filled with local minima) when combined with Sigmoid/Softmax activations. Instead, we use <strong>Cross-Entropy Loss</strong> (Log Loss), which measures the divergence between the predicted probability distribution \\( \\hat{y} \\) and the true distribution \\( y \\):
                </p>
                <div class="math-block text-center text-lg">
                    \\[ L(y, \\hat{y}) = - \\sum_{c=1}^{M} y_{o,c} \\log(\\hat{y}_{o,c}) \\]
                </div>
                <p>
                    This function heavily penalizes confident but incorrect predictions, driving the model gradients powerfully toward the correct class distribution.
                </p>
            </div>
        `
    },
    module3: {
        title: "3. Mechanics Blog",
        content: `
            <div class="prose max-w-none">
                <h2 class="text-3xl font-extrabold text-indigo-900 mb-6">Algorithm Mechanics Deep Dive</h2>
                <p class="text-lg text-gray-700 mb-8">
                    How do algorithms actually "learn"? In this module, we dissect the internal moving parts of optimization, clustering, control logic, and credit assignment within machine learning algorithms.
                </p>

                <h3>1. Gradient Descent: The Engine of Optimization</h3>
                <p>
                    Gradient Descent is a first-order iterative optimization algorithm used to find a local minimum of a differentiable function. The hypothesis function is parameterized by weights \\( \\theta \\). We define a Cost Function \\( J(\\theta) \\) that measures the error of our hypothesis.
                </p>
                <p>
                    To minimize \\( J(\\theta) \\), we compute the partial derivative (gradient) of the cost function with respect to each parameter. The weights are updated in the opposite direction of the gradient:
                </p>
                <div class="math-block text-center text-lg">
                    \\[ \\theta_j := \\theta_j - \\alpha \\frac{\\partial}{\\partial \\theta_j} J(\\theta) \\]
                </div>
                <p>
                    <strong>Key Hyperparameters:</strong><br>
                    - <strong>Learning Rate (\\( \\alpha \\)):</strong> Dictates the step size. Too small = slow convergence. Too large = overshooting the minimum and diverging.<br>
                    - <strong>Epochs:</strong> One complete pass of the training dataset through the algorithm.
                </p>

                <h3>2. K-Means Clustering: Unsupervised Partitioning</h3>
                <p>
                    K-Means partitions \\( n \\) observations into \\( k \\) clusters, where each observation belongs to the cluster with the nearest mean (centroid). It uses an Expectation-Maximization (EM) approach:
                </p>
                <ol>
                    <li><strong>Initialization:</strong> Randomly place \\( k \\) centroids in the vector space.</li>
                    <li><strong>Assignment (Expectation):</strong> Assign each data point to the nearest centroid using <em>Euclidean Distance</em>: \\( d(p,q) = \\sqrt{\\sum (q_i - p_i)^2} \\)</li>
                    <li><strong>Update (Maximization):</strong> Recalculate the position of each centroid to be the geometric mean (center of mass) of all points assigned to it.</li>
                    <li>Repeat steps 2 and 3 until centroids stop moving (convergence).</li>
                </ol>
                
                <div class="interactive-panel">
                    <h4 class="font-bold text-lg mb-4">Interactive K-Means Algorithm</h4>
                    <p class="mb-4 text-sm text-gray-600">Watch the Expectation-Maximization process in real-time. Centroids (large squares) will move towards the center of their respective clusters.</p>
                    <div class="canvas-container h-80 bg-gray-50 border rounded relative">
                        <canvas id="kmeans-canvas"></canvas>
                    </div>
                    <div class="mt-4 flex space-x-4">
                        <button id="btn-kmeans-step" class="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition">Step Algorithm</button>
                        <button id="btn-kmeans-reset" class="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition">Reset Points</button>
                    </div>
                </div>

                <h3>3. Q-Learning: Navigating the Bellman Equation</h3>
                <p>
                    Q-Learning is a model-free Reinforcement Learning algorithm. The goal is to learn a policy that tells an agent what action \\( a \\) to take under what state \\( s \\). It does this by updating a Q-Table based on the <strong>Bellman Equation</strong>:
                </p>
                <div class="math-block text-center text-lg">
                    \\[ Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ R(s,a) + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right] \\]
                </div>
                <p>
                    Where \\( \\alpha \\) is the learning rate, \\( R \\) is the immediate reward, and \\( \\gamma \\) is the discount factor (how much we care about future rewards versus immediate rewards).
                </p>
                <p>
                    <strong>Exploration vs. Exploitation:</strong> Agents use an \\( \\epsilon \\)-greedy strategy. With probability \\( \\epsilon \\), the agent takes a random action to <em>explore</em> the environment. With probability \\( 1-\\epsilon \\), it <em>exploits</em> the current Q-Table to take the best known action.
                </p>
                
                <div class="interactive-panel">
                    <h4 class="font-bold text-lg mb-4">Q-Learning Grid World</h4>
                    <p class="mb-4 text-sm text-gray-600">The agent (blue dot) learns to navigate to the goal (solid green square) while avoiding the pit (red square). The green color intensity of the cells represents the learned Q-Values.</p>
                    <div class="canvas-container h-64 w-64 mx-auto bg-gray-100 border rounded shadow-inner">
                        <canvas id="qlearning-canvas" width="256" height="256"></canvas>
                    </div>
                    <div class="mt-4 flex justify-center space-x-4">
                        <button id="btn-ql-train" class="px-4 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700 transition">Fast Train (100 Epochs)</button>
                        <button id="btn-ql-reset" class="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition">Reset Agent</button>
                    </div>
                </div>

                <h3>4. Backpropagation: The Chain Rule in Action</h3>
                <p>
                    Backpropagation is how neural networks compute gradients efficiently. When an error is calculated at the output layer, Backpropagation uses the Calculus <strong>Chain Rule</strong> to propagate this error backwards through the network layers, calculating how much each individual weight contributed to the final error.
                </p>
                <p>
                    A major challenge historically was the <strong>Vanishing Gradient Problem</strong>: when multiplying many small derivatives (using Sigmoid/Tanh activations) backward through deep networks, the gradients become exponentially small, meaning early layers barely learn. Modern architectures solve this using ReLU activations and residual connections (ResNets).
                </p>
            </div>
        `
    },
    module4: {
        title: "4. Gen AI & Transformers",
        content: `
            <div class="prose max-w-none">
                <h2 class="text-3xl font-extrabold text-indigo-900 mb-6">Generative AI & Transformer Architecture</h2>
                <p class="text-lg text-gray-700 mb-8">
                    The 2017 paper <em>"Attention Is All You Need"</em> introduced the Transformer, permanently altering the trajectory of Artificial Intelligence. This module provides a rigorous, mathematical breakdown of Large Language Models (LLMs), Transformers, and the Attention Mechanism.
                </p>

                <h3>1. The Self-Attention Mechanism</h3>
                <p>
                    Recurrent Neural Networks (RNNs) process data sequentially, making them slow and prone to forgetting long-range dependencies. Transformers process data in parallel, using <strong>Self-Attention</strong> to allow every word (token) in a sequence to look at every other word, determining which words are relevant to its current context.
                </p>
                <p>
                    For each token, the network projects its embedding into three distinct vectors using learned weight matrices: <strong>Query (Q)</strong>, <strong>Key (K)</strong>, and <strong>Value (V)</strong>.
                </p>
                <ul class="list-disc pl-6 mb-4 text-gray-800">
                    <li><strong>Query:</strong> What information am I looking for?</li>
                    <li><strong>Key:</strong> What information do I contain?</li>
                    <li><strong>Value:</strong> What is my actual semantic content?</li>
                </ul>
                <p>The Attention Score is calculated by taking the dot product of Queries and Keys, scaling them, applying Softmax to create a probability distribution, and multiplying by the Values:</p>
                <div class="math-block text-center text-lg">
                    \\[ \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V \\]
                </div>
                <p>
                    Scaling by \\( \\sqrt{d_k} \\) (the dimension of the key vectors) ensures the dot products don't grow too large, which would push the softmax function into regions with extremely small gradients.
                </p>

                <div class="interactive-panel">
                    <h4 class="font-bold text-lg mb-4">Attention Matrix Visualization</h4>
                    <p class="mb-4 text-sm text-gray-600">This simulates how a word pays attention to other words in the sentence "The bank of the river". Darker lines indicate higher attention weights. Notice how "bank" attends heavily to "river" to determine its context.</p>
                    <div class="canvas-container h-64 bg-white border rounded">
                        <canvas id="attention-canvas"></canvas>
                    </div>
                </div>

                <h3>2. The Softmax Function: Turning Raw Numbers into Probabilities</h3>
                <p>
                    At the end of a neural network layer, the output is often an array of raw, unconstrained real numbers called <strong>Logits</strong>. To use these for classification or next-token prediction, we must convert them into a valid probability distribution (all values between 0 and 1, summing to 1). This is the role of the Softmax function.
                </p>
                <div class="math-block text-center text-lg">
                    \\[ \\sigma(\\mathbf{z})_i = \\frac{e^{z_i}}{\\sum_{j=1}^{K} e^{z_j}} \\]
                </div>
                <p>
                    By exponentiating the logits (\\( e^{z_i} \\)), Softmax accomplishes two things:
                    <ol class="list-decimal pl-6 mt-2 text-gray-800">
                        <li>It makes all values strictly positive.</li>
                        <li>It exponentially exaggerates differences. A slightly larger logit becomes a significantly larger probability.</li>
                    </ol>
                </p>

                <div class="interactive-panel">
                    <h4 class="font-bold text-lg mb-4">Interactive Softmax Calculator</h4>
                    <p class="mb-4 text-sm text-gray-600">Adjust the raw Logit values below. Observe how the exponential function transforms them, and how the final Probabilities always sum to 100%. Notice how raising one logit suppresses the probabilities of all others.</p>
                    
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mt-6">
                        <div class="space-y-4" id="softmax-sliders">
                            <!-- Sliders injected by JS -->
                        </div>
                        <div class="canvas-container h-48 bg-slate-50 border rounded flex items-end justify-around p-4" id="softmax-bars">
                            <!-- Bars injected by JS -->
                        </div>
                    </div>
                </div>

                <h3>3. Autoregressive Generation and LLMs vs. SLMs</h3>
                <p>
                    Models like GPT (Generative Pre-trained Transformer) are <strong>Autoregressive</strong>. They predict the probability distribution of the <em>next</em> token based on the sequence of <em>all previous</em> tokens. Once a token is generated, it is appended to the input sequence, and the process repeats.
                </p>
                <div class="bg-indigo-50 p-6 rounded my-6 border-l-4 border-indigo-500">
                    <h4 class="font-bold mb-2 text-indigo-900">LLMs vs SLMs (Small Language Models)</h4>
                    <p class="mb-2 text-indigo-800"><strong>Large Language Models (e.g., GPT-4, Llama 3 70B):</strong> Feature tens to hundreds of billions of parameters. They exhibit emergent capabilities (zero-shot reasoning, complex logic) but require massive VRAM and distributed GPU clusters to run.</p>
                    <p class="text-indigo-800"><strong>Small Language Models (e.g., Phi-3, Llama 3 8B):</strong> Feature 1 to 8 billion parameters. Through highly curated training data (often synthetic data generated by larger models), they achieve remarkably high performance on specific tasks while being able to run locally on consumer hardware (laptops, phones).</p>
                </div>
            </div>
        `
    }
};