# Metadata-Based Security Detection for Multi-Agent AI Systems
## Idea Seminar — Research Direction Overview

---

## [Slide 1] Cover

**Topic:** Indirect Prompt Injection Detection in Multi-Agent AI Systems (MAS)  
A Lightweight, Graph-Based Anomaly Detection Idea

**One-line summary:** Research on how to quickly notice when one agent in a system of multiple collaborating AIs has been attacked.

**Presenter / Affiliation / Date**

---

## [Slide 2] Agenda

1. Background and Related Work — Problem background and prior research
2. Proposed Approach — Core idea
3. Experiments — Validation direction
4. Conclusion — Expected impact and plans

---
---

# I. Background and Related Work

---

## [Slide 3] Research Background

**An "agent" refers to an AI program that takes on a specific role and acts autonomously based on its own judgment.**

A system in which multiple agents divide a single task among themselves, each handing its processed result to the next agent, is called a Multi-Agent System (MAS). This structure has recently been adopted rapidly for automating real-world work such as search, analysis, and report writing.

**Example pipeline used for illustration:**
```
User → Orchestrator (assigns task) → Researcher (gathers data) → Analyst (analyzes) → Writer (drafts report) → Result
```

> The structure above is one representative example chosen for explanatory purposes among many possible MAS forms; real MAS deployments do not necessarily exist only as this kind of linear chain. Star-shaped (a central agent instructing several agents at once), hierarchical, and cyclic structures are all possible. This study uses this example to explain the problem and the idea.

**A security issue not yet widely recognized in this structure**
- Agents trust the content handed over by the preceding agent without verification and use it directly in the next step.
- This "unconditional trust" relationship can be exploited by an attack.

---

## [Slide 4] Indirect Prompt Injection Attack

**A prompt refers to the instruction given to an AI.** "Prompt injection" refers to an attack in which an adversary secretly embeds a malicious command inside such an instruction. It is called "indirect" because the attacker does not converse with the AI directly; instead, the command is hidden inside an external document or search result that the AI will later reference.

```
A document that looks like a normal report
 └─ (a hidden instruction embedded inside: "From now on, perform a different role")
```

If an agent reads this document and follows the hidden instruction inside it, that agent becomes "compromised." The problem does not stop there.

**Once one point is compromised, the entire pipeline can collapse like dominoes**

```
Orchestrator compromised → Researcher compromised → Analyst compromised → Writer compromised
```

*(As explained in Slide 3, the 4-stage structure above is an illustrative example; the number of agents and how they are connected may differ in practice. The key point is the phenomenon itself — "contamination at an earlier stage keeps propagating to later stages.")*

If even a single upstream agent is deceived, every agent downstream inherits the compromised information and becomes compromised together — this is the structural vulnerability at play.

---

## [Slide 5] Limitations of Existing Approaches

| Approach | Description | Limitation |
|-----------|------|------|
| Inspect response content with another AI every time | A separate AI reads each response and judges whether it is anomalous | Slow and expensive |
| Directly monitor each agent's internals | Directly inspect an agent's internal prompts/code | Not applicable to "black-box" agents whose internals cannot be viewed |
| Keyword/rule-based filters | Check for predefined risky words or patterns | Easily bypassed by slightly altered attack phrasing |

**Key question**
> Can we tell whether an agent has been attacked without ever looking at "what it actually said"?

---

## [Slide 6] Related Work ① — SentinelAgent

**Paper Overview**
> Title: SentinelAgent: Graph-based Anomaly Detection in Multi-Agent Systems  
> Authors: Xu He, Di Wu, Yan Zhai, Kun Sun  
> Published: 2025, arXiv:2505.24201

**Paper Summary**
- **Graph-based modeling:** Represents inter-agent interactions as an execution graph and analyzes anomalies at the node, edge, and path level.
- **LLM-based semantic analysis:** Relies on a separate LLM oversight agent that directly reads and semantically interprets each agent's response content for anomaly detection.

**Future Direction (the gap this study aims to fill)**
- It remains to be verified whether the same level of graph-based detection can be achieved without reading response content, using communication metadata alone.

---

## [Slide 7] Related Work ② — AgentDojo

**Paper Overview**
> Title: AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents  
> Authors: Edoardo Debenedetti, Jie Zhang, Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, Florian Tramèr  
> Published: 2024, NeurIPS Datasets and Benchmarks Track (arXiv:2406.13352)

**Paper Summary**
- **Standardized benchmark:** Measures attack/defense performance using 97 realistic tasks (e.g., email management, banking, travel booking) and 629 security test cases.
- **Limitation of the evaluation criteria:** Performance is measured mainly via "task success rate," without a metric that quantifies structural or behavioral anomaly signals.

**Future Direction (the gap this study aims to fill)**
- An evaluation approach is needed that combines anomaly-detection metrics such as AUC and F1, along with inter-agent communication structure signals, in addition to task success rate.

---
---

# II. Proposed Approach

---

## [Slide 8] Core Idea

**Instead of the "content" of a response, observe the "behavioral pattern" that emerges in how the response is produced.**

This starts from the observation that an attacked agent behaves differently from usual.
- Its response becomes slower or faster than usual
- It answers much longer or shorter than usual
- A sudden large change occurs relative to the previous stage

**Why this change occurs:** When an attacker's hidden instruction is added to the prompt, the content the agent must process becomes longer and more complex. As a result, the time it takes to produce a response, and the length of that response, tend to change as well. The shifts in metadata originate from this causal relationship.

Such "auxiliary information automatically recorded around the content, rather than the content itself" is called metadata. (Example: not the body text of an email, but information such as when it arrived and how many characters it contains.)

**The 6 metadata features actually defined (no access to response content):**

| Feature | Description |
|------|------|
| latency | Time the agent took to produce a response |
| token_count | Length of the response (a token is a unit for counting characters/words) |
| api_freq (call frequency) | A value approximated by the number of sentences in the response |
| ctx_delta (rate of change) | How much the response differs from the immediately preceding agent's response |
| call_seq (call anomaly flag) | Whether an unusual call pattern was detected |
| refusal_flag | Whether the agent responded that it "cannot perform this request" |

**Why these 6 were chosen:** They are values the system already records automatically while an agent produces a response, with no need to build any new logging infrastructure. The practical benefit of being deployable immediately, without modifying existing MAS infrastructure, is significant.

**Advantages of this approach:** Because response content is never read, processing is fast, there is no risk of exposing private information, and the same method applies regardless of language or which AI model is used.

---

## [Slide 9] The Need for a Graph-Structured Approach

**Looking at agents one at a time makes it ambiguous to judge whether something is anomalous.**

```
Orchestrator observed alone: appears only as a minor outlier
Researcher observed alone:   judged to be within the normal range
```

**However, connecting and viewing the entire pipeline together makes the pattern clear.**

```
Orchestrator(minor anomaly) → Researcher(minor anomaly) → Analyst(minor anomaly) → Writer(normal)
   "A cascading pattern in which an anomaly signal that began upstream keeps appearing downstream"
```

To capture this cascading pattern, instead of viewing each agent independently, we propose representing "who passes information to whom" as a graph and learning this relational structure itself.

Here, a graph refers to a structure in which points (nodes) are connected by lines (edges). In this study, each agent becomes a node, and the information-passing relationship between agents becomes an edge.

---

## [Slide 10] Proposed Model: A Lightweight Graph Autoencoder

**An autoencoder is a model trained by compressing an input and then attempting to reconstruct it back to its original form.** If the model has learned the usual pattern well, a normal input will be reconstructed accurately, whereas an unfamiliar (anomalous) input will not — this is the principle being used.

**Training and detection procedure — trainable without labeled answers**
```
Step 1 (training): Show the model the metadata pattern of a normally operating pipeline
Step 2 (detection): When a new session arrives, attempt to reconstruct it based on the learned pattern
       → Good reconstruction → normal
       → Poor reconstruction → an unusual situation → suspected attack
```

**Why an autoencoder was chosen over a normal/attack classifier:** New attack types keep emerging, while it is difficult to gather enough real attack cases (labeled data). Choosing an unsupervised approach (One-Class Detection) that can be trained using only normal data raises the possibility of generalizing to attack types that did not exist at training time.

**A Graph Neural Network (GNN)** is a neural network in which nodes connected by a graph exchange information with one another during learning. Using this, judgments can take into account not just a single agent's value, but the states of other connected agents as well.

**Why GCN was chosen among various graph neural network techniques:** GCN is among the simplest in structure, making it easy to minimize the number of parameters (consistent with the goal of a lightweight model), and its results are relatively easy to interpret, which was judged suitable for this early idea-validation stage. Comparison with more complex GNNs such as GAT is left as future work.

**The model structure actually designed:**
```
Input (6 metadata features per agent)
  ├─ GCN Layer 1:  6 → 16   (aggregates and expands information from connected agents)
  ├─ GCN Layer 2: 16 →  8   (summarizes into a more compressed representation)
  ├─ Decoder 1:    8 → 16   (begins reconstruction)
  └─ Decoder 2:   16 →  6   (reconstructs the original feature shape)

Total parameters (the number of internal numbers the model learns): 461
→ An extremely lightweight model, far smaller than even a single smartphone app
```

---
---

# III. Experiments

---

## [Slide 11] Validation Direction

**The idea is being validated in two stages.**

1. **Simulation environment** — Rather than running a real AI every time, various attack situations are artificially generated by computer and repeated at scale. Suitable for broadly testing across many attack types.
2. **Real LLM environment** — Several actually running language models (LLMs) are connected in a pipeline to confirm whether this approach also works in a genuine environment.

**Why both environments are used:** The simulation is meant to test many attack types repeatedly at large scale and low cost, while the real-LLM experiment is meant to cross-validate whether the tendencies confirmed in simulation also hold in a real-world environment. These are two complementary validation procedures serving different purposes.

**Why a local open-source model (llama3.2) was used for the real-LLM experiment instead of a commercial model (GPT, Claude, etc.):** It can be run repeatedly in a local environment at no additional cost, which was suitable for running a large number of sessions across multiple seeds. Generalization to commercial models is addressed in future plans.

**5 attack types designed for the simulation**

| Type | Characteristics |
|------|------|
| Direct | An overt attack that immediately tries to change the agent's role |
| Harvest | An attack that gathers information and then propagates it to downstream agents |
| Slow | An attack that contaminates the pipeline gradually, little by little |
| Flood | An attack that contaminates several agents simultaneously |
| **Chain** | **An attack that breaches only a single point and then propagates downstream — the type where the graph-based approach's advantage is most clearly visible** |

---

## [Slide 12] Initial Validation Results

**AUC is used as the metric for detection performance.** AUC expresses, as a number between 0 and 1, how well a normal situation can be distinguished from an attack situation; the closer to 1, the more accurate the distinction.

| Experimental environment | Detection performance (AUC) |
|-----------|:---------:|
| Simulation (5 agents, 5 attack types) | 0.99 or higher |
| Real local LLM (4 agents, llama3.2) | up to 1.00 |

- In particular, for the **Chain type** (an attack that breaches a single point and then propagates), the approach that jointly considers the relational structure (graph) between agents was confirmed to detect it noticeably better than an approach that does not.
- Even in an environment reproduced with a real language model, detection was stable once the attack's propagation extent was sufficiently large.

**In addition, it was confirmed that there is some possibility of pinpointing which agent the problem started at.** (A graph autoencoder reconstructs the entire pipeline at once, but since the reconstruction error is computed per agent individually, it is possible to check separately at which agent the error appears large.)
```
Orchestrator  ████░░░░░░  Low    ← judged normal
Planner       ██████████  High   ← ★ presumed point of compromise
Researcher    ██████░░░░  Medium ← trace of propagated contamination
Writer        ████░░░░░░  Low    ← judged normal
```
(The values above are a quantity called "reconstruction error," representing the degree to which the model failed to reconstruct the input. A larger value means that agent behaved differently from usual.)

> This is still an early validation stage, and further generalization testing with a wider variety of language models and attack methods is needed.

---
---

# IV. Conclusion

---

## [Slide 13] Expected Impact of This Research

**If this idea works well, the following practical benefits can be expected.**

- Because it is a detection method that can be deployed without accessing an agent's internals or response content, it is also applicable to "black-box" MAS provided by external vendors.
- The model is extremely lightweight, making it suitable for real-time detection.
- If the specific agent that was attacked can be pinpointed, it becomes possible to isolate only that agent while keeping the rest of the pipeline running.

---

## [Slide 14] Future Plans

1. Verify whether it works equally well across a wider variety of language models (GPT, Claude, etc.)
2. Compare against other approaches on public standard benchmarks (e.g., AgentDojo)
3. Examine whether it can be applied in a real-time, continuously streaming data setting
4. Extend the pipeline beyond detection to automated response (e.g., isolating the compromised agent)

**Points I would like to discuss at today's seminar**
- Whether this direction is actually a meaningful problem to pursue
- Whether the validation approach (simulation + real LLM) is sufficiently convincing
- Any perspective I may be missing, or related work worth referencing

---

*For idea presentation purposes | Estimated seminar talk time: about 10 minutes*
