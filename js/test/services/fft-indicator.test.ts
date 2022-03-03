import app from '../../src/app';

describe('\'fft-indicator\' service', () => {
  it('registered the service', () => {
    const service = app.service('fft-indicator');
    expect(service).toBeTruthy();
  });
});
