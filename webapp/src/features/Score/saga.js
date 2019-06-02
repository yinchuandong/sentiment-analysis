import {
    call, put, takeLatest, fork, select,
  } from 'redux-saga/effects'
  import * as Types from './actionType'
  import Actions from './action'
  import ScoreApi from '../../apis/scoreApi'
  
  export function* workDoScore({ payload }) {
    const req = {
      score: payload,
    }
    try {
      console.log('saga: DO_SCORE api:', req)
    //   const response = yield call(ScoreApi.doScore, req)// use asynchronous function 调用异步函数
    //   if (response.data.code === 0) {
        yield put(Actions.doScoreSuccess({score:0.73}))// like dispatch
    //   } else {
        // yield put(Actions.failure(response.data.message))
    //   }
    } catch (error) {
      yield put(Actions.doScoreFailure(error))
    }
  }
  
  function* watchDoScore() {
    console.log('watching DO_SDO_SCORE_REQUESTCORE')
    yield takeLatest(Types.DO_SCORE_REQUEST, workDoScore)// listen action，if it dispatch "GETLIST_REQUEST",then trigger function "workGetlist"
  }
  
  export default [fork(watchDoScore)]// Non-blocking task invocation mechanism 非阻塞任务调用机制
  