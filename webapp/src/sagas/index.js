import { all } from 'redux-saga/effects'
import { ScoreSaga } from '../features/Score'

// combine all the sagas
export default function* rootSaga() {
  yield all([
    ...ScoreSaga,
  ])
}
