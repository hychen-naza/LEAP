for env in GoToObjMazeS5 GoToObjMazeS6 GoToObjMazeS7
do
    python ./scripts/make_agent_demos.py --env BabyAI-${env}-v0 --episodes 500
done
