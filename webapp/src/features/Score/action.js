import { createAction } from '../../utils/ActionUtil'
import * as Types from './actionType'

// define actions
const Actions = {
  changeName: name => createAction(Types.CHANGE_NAME, name),
  doScoreRequest: name => createAction(Types.DO_SCORE_REQUEST, name),
  doScoreSuccess: score => createAction(Types.DO_SCORE_SUCCESS, score),
  doScoreFailure: error => createAction(Types.DO_SCORE_FAILURE, error),
  resetScore: () => createAction(Types.RESET_SCORE),
}

export default Actions
