/** @type {import('ts-jest/dist/types').InitialOptionsTsJest} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testTimeout: 30000,
  globals: {
    'ts-jest': {
      diagnostics: false,
      tsconfig: 'test/tsconfig.json'
    }
  }
};
