# API Scheme
A naming scheme for our API. The goal is to make the grammar of the entire API consistent.

## Rules

### Node Rules
All node names must be:
1. Nouns
2. Singular

### Coordinates Rules
All Coordinates names must be:
1. Nouns

### Core Directory Rules
Where a "Core Directory" is located inside podpac/core/*
All directory names must be:
1. Nouns
2. Pluarl

### Package Rules:

Where a "Package" is loaded by podpac.*

All package names must be:
1. Nouns
2. Pluarl
*AND*
3. All packages must match their corresponding `podpac.core.*` directory name.