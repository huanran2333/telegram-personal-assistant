import { describe, expect, it, vi } from 'vitest';
import { SubmissionKeys } from './api';

describe('durable HTTP submission identity', () => {
  it('reuses the same key after an unknown response and changes it for a changed request', () => {
    vi.stubGlobal('crypto', { randomUUID: vi.fn()
      .mockReturnValueOnce('key-a').mockReturnValueOnce('key-b').mockReturnValueOnce('key-c') });
    const keys = new SubmissionKeys();
    expect(keys.for('todo-create', { title: 'A' })).toBe('key-a');
    expect(keys.for('todo-create', { title: 'A' })).toBe('key-a');
    expect(keys.for('todo-create', { title: 'B' })).toBe('key-b');
    keys.clear('todo-create');
    expect(keys.for('todo-create', { title: 'B' })).toBe('key-c');
    vi.unstubAllGlobals();
  });
});
