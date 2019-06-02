import React, { Component } from "react";
import { Route, Switch } from "react-router";
import ScorePage from "../Score";
import './App.less'




class App extends Component {
  render() {
    return (
      <div className="App">
        <Switch>
          <Route exact path="/" component={ScorePage} />
          <Route exact path="/score" component={ScorePage} />
          <Route exact path="/score/:key" component={ScorePage} />
        </Switch>
      </div>
    );
  }
}

export default App;
