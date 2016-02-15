'use strict';

/* jshint ignore:start */
var expect = require('expect.js');
/* jshint ignore:end */
var _ = require('lodash');
var nj = require('../../src');

describe('ones', function () {
    it('should exist', function () {
        expect(nj.ones).to.be.ok();
    });

    it('can generate a vectors', function(){
        expect(nj.ones(0).tolist()).to.eql([]);
        expect(nj.ones(2).tolist()).to.eql([1,1]);
        expect(nj.ones([2]).tolist()).to.eql([1, 1]);
    });

    it('can generate matrix', function(){
        expect(nj.ones([2,2]).tolist())
            .to.eql([[1, 1], [1, 1]]);
    });

    it('should accept a dtype', function(){
        expect(nj.ones(0, 'uint8').dtype).to.be('uint8');
    });
});