# agent-skills

Shared workflow assets for importing into project repos under `.agents/`. After import, start with `.agents/AGENTS.md` and use `$work`.

```bash
git remote add agent-skills https://github.com/wmentrekin/agent-skills.git
git subtree add --prefix=.agents agent-skills main --squash
```

```bash
git subtree pull --prefix=.agents agent-skills main --squash
```
