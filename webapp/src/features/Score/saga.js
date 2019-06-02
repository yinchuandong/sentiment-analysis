import { call, put, takeLatest, fork, select } from 'redux-saga/effects'
import * as Types from './actionType'
import Actions from './action'
import ScoreApi from '../../apis/scoreApi'

export function* workDoScore({ payload }) {
  const req = {
    text: [payload]
  }
  try {
    console.log('saga: DO_SCORE api:', req)
    const response = yield call(ScoreApi.predict, req)
    if (response.data.status === 'success') {
      console.log(response.data.status)
      yield put(Actions.doScoreSuccess({ score: response.data.score[0] })) // like dispatch
    } else {
      yield put(Actions.doScoreFailure(response.data.message))
    }
  } catch (error) {
    console.log(error)
    yield put(Actions.doScoreFailure(error))
  }
}

function* watchDoScore() {
  console.log('watching DO_SDO_SCORE_REQUESTCORE')
  yield takeLatest(Types.DO_SCORE_REQUEST, workDoScore)
}

export default [fork(watchDoScore)]
