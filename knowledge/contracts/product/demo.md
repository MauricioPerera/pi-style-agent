---
type: 'Task Contract'
title: 'Offline demo completes'
description: 'Keep the existing offline agent demo executable.'
tags: ['demo', 'offline', 'adoption']
task: offline-demo-completes
intent: 'Preserve the offline demo entry point.'
target: 'demo.py'
signature: 'def main():'
test_command: 'python -m unittest tests/test_demo.py'
test_runner: python-unittest
oracle_files: ['tests/test_demo.py', 'tests/__init__.py']
budget:
  cyclomatic_max: 12
  nesting_max: 3
tests: 'tests/test_demo.py'
tests_sha256: '379592b1c361f4d90dbbf8c5953425d4c3db50237ebf16900960a5289d9e3ba1'
touch_only: ['demo.py']
deps_allowed: []
forbids: ['network']
---

# Offline demo completes

## Intent

The [adoption record](../../adoption.md) describes how this existing smoke test is selected by KDD.

## Interface

Run the existing offline demo through the declared test.

## Invariants

- The stub provider runs the complete demo.
- A demo assertion or unhandled error must fail the smoke test.

## Examples

- The current demo completes -> the test passes.
- A deliberate failure in demo.main -> the same test fails.

## Do / Don't

- DO execute the existing smoke test.
- DON'T replace its assertions with another oracle.

## Tests

tests/test_demo.py invokes demo.py through the stub provider.

## Constraints

PARAR y reportar si la adopción requiere editar las pruebas originales.
