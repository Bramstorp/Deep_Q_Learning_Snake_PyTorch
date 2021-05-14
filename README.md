# Deep_Q_Learning_Snake_PyTorch
Reinforcement Learning and Deep Q Learning snake with pygames and pytorch

# Run Training

- Python3 agent.py

# Reward System
- Eat food: +10
- Game Over: -10
- Else: 0

# Actions System

-[1,0,0]: Forward

-[0,1,0]: Right turn

-[0,0,1]: Left turn

# State System

States: 11

Dangers: Left, Forward, Left

Direction: Up, Down, Left, Right

Food: Up, Down, Left, Right

# Q Learning

Q Value = Quality of action

0. Init Q Value (init model)

1. Choose Action (model prediction of state)

2. Perform Action

3. Measure Reward

4. Update Q Value and train model 

# Q Update Rule

Q = model.predict(State0)

New Q = Reward + Gamme Value * MAX(Q(state1))

Loss Function:

loss = (New Q - Q)*2
