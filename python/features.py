import numpy as np
import pandas as pd
import math
from datetime import date
from sklearn.decomposition import PCA


def gender_idx(gender):
    if gender == 'FEMALE':
        return 1
    elif gender == 'MALE':
        return 2
    return 0


def str_to_date(str_date):
    y=int(str_date.split('-')[0])
    m=int(str_date.split('-')[1])
    d=int(str_date.split('-')[2])
    return date(y, m, d)


def date_diff(y1, m1, d1, y2, m2, d2):
    d1 = date(y1, m1, d1)
    d2 = date(y2, m2, d2)
    delta = d2 - d1
    return delta.days


def make_one_hot(df, feature, values=None, test_columns=None, use_threshold=False):
    dummy_df = pd.get_dummies(df[feature], prefix=feature)
    if values is not None:
        dummy_df = dummy_df[[feature + '_' + value for value in values]]

    if test_columns is not None:
        columns_to_drop = []
        for column in list(dummy_df.columns.values):
            if column not in test_columns:
                columns_to_drop.append(column)
        dummy_df = dummy_df.drop(columns_to_drop, axis=1)

    # if use_threshold:
    #    threshold = 0.001
    #    n_rows = len(dummy_df.index)
    #    rare_columns = []
    #    for column in list(dummy_df.columns.values):
    #        val_count = len(dummy_df[dummy_df[column] == 1].index)
    #        if threshold * n_rows > val_count:
    #            rare_columns.append(column)
    #    dummy_df = dummy_df.drop(rare_columns, axis=1)

    return pd.concat((df, dummy_df), axis=1)


def do_pca(x):
    pca = PCA()
    pca.fit(x)
    print('variance_ratio: ', pca.explained_variance_ratio_)

    n_components = 0
    acc = 0.0
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        acc += ratio
        if acc > 0.9999:
            n_components = i
            break

    pca = PCA(n_components=n_components)
    pca.fit(x)
    print('variance_ratio: ', pca.explained_variance_ratio_)
    return pca


def remove_sessions_columns(x, columns):
    new_columns_idx = []
    for i, column in enumerate(columns):
        if not column.startswith('s_'):
            new_columns_idx.append(i)
    return x[:, new_columns_idx]


def remove_no_sessions_columns(x, columns):
    new_columns_idx = []
    for i, column in enumerate(columns):
        if column.startswith('s_'):
            new_columns_idx.append(i)
    return x[:, new_columns_idx]


def divide_by_has_sessions(count_all_col_idx, x, y):
    rows_filter_sessions = (x[:, count_all_col_idx] == 2014)
    rows_filter_no_sessions = (x[:, count_all_col_idx] != 2014)

    x_sessions = x[rows_filter_sessions, :]
    x_no_sessions = x[rows_filter_no_sessions, :]
    if y is None: return x_sessions, x_no_sessions

    y_sessions = y[rows_filter_sessions, ]
    y_no_sessions = y[rows_filter_no_sessions, ]
    return x_sessions, y_sessions, x_no_sessions, y_no_sessions


def sync_columns(x, x_columns, no_remove_columns):
    x = np.copy(x)
    test_columns_indices_to_remove = []
    new_columns = []
    for i, column in enumerate(x_columns):
        if column not in no_remove_columns:
            test_columns_indices_to_remove.append(i)
        else:
            new_columns.append(column)
    x = np.delete(x, test_columns_indices_to_remove, 1)
    return x, new_columns


def add_features(data_df, test_columns):
    print('add_features <<')
    data_df = data_df.copy()

    data_df['age'] = data_df.apply(
        lambda r: 2015 - r['age'] if (np.isfinite(r['age']) & (r['age'] >= 1900) & (r['age'] < 2000)) else r['age'], axis=1)
    data_df['has_age'] = data_df.apply(lambda r: 0 if pd.isnull(r['age']) or r['age'] <= 16 or r['age'] >= 80 else 1, axis=1)

    data_df['age_imp'] = data_df.apply(lambda r: r['age'] if (np.isfinite(r['age']) & (r['age'] >= 16) & (r['age'] < 80)) else -1, axis=1)

    # mean_age = data_df[np.isfinite(data_df['age']) & (data_df['age'] < 100)]['age'].mean()
    # data_df['age_imp_mean'] = data_df.apply(
    #    lambda r: r['age'] if (np.isfinite(r['age']) & (r['age'] >= 16) & (r['age'] < 80)) else mean_age, axis=1)

    data_df['day_account_created'] = data_df.apply(lambda r: int(str(r['date_account_created']).split('-')[2]), axis=1)
    data_df['month_account_created'] = data_df.apply(lambda r: int(str(r['date_account_created']).split('-')[1]), axis=1)
    data_df['year_account_created'] = data_df.apply(lambda r: int(str(r['date_account_created']).split('-')[0]), axis=1)
    data_df['day_of_week'] = data_df.apply(lambda r: str_to_date(r['date_account_created']).weekday(), axis=1)
    data_df['day_of_year'] = data_df.apply(lambda r: 12 * r['month_account_created'] + r['day_account_created'], axis=1)
    data_df = data_df.drop(['day_account_created', 'month_account_created'], axis=1)

    data_df['day_first_active'] = data_df.apply(lambda r: int(str(r['timestamp_first_active'])[6:8]), axis=1)
    data_df['month_first_active'] = data_df.apply(lambda r: int(str(r['timestamp_first_active'])[4:6]), axis=1)
    data_df['year_first_active'] = data_df.apply(lambda r: int(str(r['timestamp_first_active'])[0:4]), axis=1)

    data_df['day_of_week_first_active'] = data_df.apply(
        lambda r: date(r['year_first_active'], r['month_first_active'], r['day_first_active']).weekday(), axis=1)

    data_df['day_of_year_first_active'] = data_df.apply(
        lambda r: 12 * r['month_first_active'] + r['day_first_active'], axis=1)
    # data_df = data_df.drop(['year_account_created', 'year_first_active'], axis=1)

    data_df = make_one_hot(data_df, 'language', test_columns=test_columns)
    data_df = make_one_hot(data_df, 'gender', test_columns=test_columns)
    data_df = make_one_hot(data_df, 'first_device_type', test_columns=test_columns)
    data_df = make_one_hot(data_df, 'affiliate_channel', test_columns=test_columns)
    data_df = make_one_hot(data_df, 'affiliate_provider', test_columns=test_columns)
    data_df = make_one_hot(data_df, 'first_affiliate_tracked', test_columns=test_columns)
    data_df = make_one_hot(data_df, 'signup_app', test_columns=test_columns)
    data_df = make_one_hot(data_df, 'signup_method', test_columns=test_columns)
    data_df = make_one_hot(data_df, 'first_browser', test_columns=test_columns)
    data_df = make_one_hot(data_df, 'signup_flow', test_columns=test_columns)

    drop_columns = ['date_account_created', 'timestamp_first_active', 'date_first_booking', 'gender', 'age',
                    'signup_method', 'language', 'affiliate_channel', 'affiliate_provider',
                    'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser' ]

    data_df = data_df.drop(drop_columns, axis=1)

    print('add_features >>')

    return data_df


def sessions_has_column_feature(data_df, sessions_df, column, feature):
    sessions_df=sessions_df[['user_id', column]]
    df_sessions_action_type = sessions_df[sessions_df[column] == feature]

    df_sessions_action_feature_counts = df_sessions_action_type.groupby(['user_id', column]).size().reset_index()
    df_sessions_action_feature_counts.columns = ['id', column, 'count']

    df_action_type_counts = pd.merge(data_df, df_sessions_action_feature_counts, on='id', how='left')[['id', 'count',
                                                                                                       's_count_all']]

    df_action_type_counts['count'] = df_action_type_counts.apply(
        lambda r : r['count'] if not math.isnan(r['count']) else (-1 if r['s_count_all'] == -1 else 0), axis=1)

    return df_action_type_counts['count']


def sessions_has_action_detail(data_df, sessions_df, action_detail):
    return sessions_has_column_feature(data_df, sessions_df, 'action_detail', action_detail)


def sessions_has_action_type(data_df, sessions_df, action_type):
    return sessions_has_column_feature(data_df, sessions_df, 'action_type', action_type)


def sessions_has_action(data_df, sessions_df, action):
    return sessions_has_column_feature(data_df, sessions_df, 'action', action)


def add_sessions_features(data_df, sessions_df):

    print('add_sessions_data <<')

    sessions_actions_count_df=pd.DataFrame({'s_count_all' : sessions_df.groupby(['user_id']).size()}).reset_index()
    sessions_actions_count_df.columns=['id', 's_count_all']
    data_df = pd.merge(data_df, sessions_actions_count_df, on='id', how='left')
    data_df['s_count_all'] = data_df.apply(lambda r: r['s_count_all'] if (r['s_count_all'] > 0) else -1, axis=1)
    #return data_df

    data_df['s_has_action_show'] = sessions_has_action(data_df, sessions_df, 'show')
    data_df['s_has_action_index'] = sessions_has_action(data_df, sessions_df, 'index')
    data_df['s_has_action_search_results'] = sessions_has_action(data_df, sessions_df, 'search_results')
    data_df['s_has_action_personalize'] = sessions_has_action(data_df, sessions_df, 'personalize')
    data_df['s_has_action_search'] = sessions_has_action(data_df, sessions_df, 'search')
    data_df['s_has_action_ajax_refresh_subtotal'] = sessions_has_action(data_df, sessions_df, 'ajax_refresh_subtotal')
    data_df['s_has_action_update'] = sessions_has_action(data_df, sessions_df, 'update')
    data_df['s_has_action_similar_listings'] = sessions_has_action(data_df, sessions_df, 'similar_listings')
    data_df['s_has_action_social_connections'] = sessions_has_action(data_df, sessions_df, 'social_connections')
    data_df['s_has_action_reviews'] = sessions_has_action(data_df, sessions_df, 'reviews')
    data_df['s_has_action_active'] = sessions_has_action(data_df, sessions_df, 'active')
    data_df['s_has_action_similar_listings_v2'] = sessions_has_action(data_df, sessions_df, 'similar_listings_v2')
    data_df['s_has_action_lookup'] = sessions_has_action(data_df, sessions_df, 'lookup')
    data_df['s_has_action_create'] = sessions_has_action(data_df, sessions_df, 'create')
    data_df['s_has_action_dashboard'] = sessions_has_action(data_df, sessions_df, 'dashboard')
    data_df['s_has_action_header_userpic'] = sessions_has_action(data_df, sessions_df, 'header_userpic')
    data_df['s_has_action_collections'] = sessions_has_action(data_df, sessions_df, 'collections')
    data_df['s_has_action_edit'] = sessions_has_action(data_df, sessions_df, 'edit')
    data_df['s_has_action_campaigns'] = sessions_has_action(data_df, sessions_df, 'campaigns')
    data_df['s_has_action_track_page_view'] = sessions_has_action(data_df, sessions_df, 'track_page_view')
    data_df['s_has_action_unavailabilities'] = sessions_has_action(data_df, sessions_df, 'unavailabilities')
    data_df['s_has_action_qt2'] = sessions_has_action(data_df, sessions_df, 'qt2')
    data_df['s_has_action_notifications'] = sessions_has_action(data_df, sessions_df, 'notifications')
    data_df['s_has_action_confirm_email'] = sessions_has_action(data_df, sessions_df, 'confirm_email')
    data_df['s_has_action_requested'] = sessions_has_action(data_df, sessions_df, 'requested')
    data_df['s_has_action_identity'] = sessions_has_action(data_df, sessions_df, 'identity')
    data_df['s_has_action_ajax_check_dates'] = sessions_has_action(data_df, sessions_df, 'ajax_check_dates')
    data_df['s_has_action_show_personalize'] = sessions_has_action(data_df, sessions_df, 'show_personalize')
    data_df['s_has_action_ask_question'] = sessions_has_action(data_df, sessions_df, 'ask_question')
    data_df['s_has_action_listings'] = sessions_has_action(data_df, sessions_df, 'listings')
    data_df['s_has_action_authenticate'] = sessions_has_action(data_df, sessions_df, 'authenticate')
    data_df['s_has_action_calendar_tab_inner2 '] = sessions_has_action(data_df, sessions_df, 'calendar_tab_inner2')
    data_df['s_has_action_travel_plans_current'] = sessions_has_action(data_df, sessions_df, 'travel_plans_current')
    data_df['s_has_action_edit_verification'] = sessions_has_action(data_df, sessions_df, 'edit_verification')
    data_df['s_has_action_ajax_lwlb_contact'] = sessions_has_action(data_df, sessions_df, 'ajax_lwlb_contact')
    data_df['s_has_action_other_hosting_reviews_first'] = sessions_has_action(data_df, sessions_df, 'other_hosting_reviews_first')
    data_df['s_has_action_recommendations'] = sessions_has_action(data_df, sessions_df, 'recommendations')
    data_df['s_has_action_manage_listing'] = sessions_has_action(data_df, sessions_df, 'manage_listing')
    data_df['s_has_action_click'] = sessions_has_action(data_df, sessions_df, 'click')
    data_df['s_has_action_complete_status'] = sessions_has_action(data_df, sessions_df, 'complete_status')
    data_df['s_has_action_ajax_photo_widget_form_iframe'] = sessions_has_action(data_df, sessions_df, 'ajax_photo_widget_form_iframe')
    data_df['s_has_action_payment_instruments'] = sessions_has_action(data_df, sessions_df, 'payment_instruments')
    data_df['s_has_action_message_to_host_focus'] = sessions_has_action(data_df, sessions_df, 'message_to_host_focus')
    data_df['s_has_action_verify'] = sessions_has_action(data_df, sessions_df, 'verify')
    data_df['s_has_action_payment_methods'] = sessions_has_action(data_df, sessions_df, 'payment_methods')
    data_df['s_has_action_cancellation_policies'] = sessions_has_action(data_df, sessions_df, 'cancellation_policies')
    data_df['s_has_action_callback'] = sessions_has_action(data_df, sessions_df, 'callback')
    data_df['s_has_action_settings'] = sessions_has_action(data_df, sessions_df, 'settings')
    data_df['s_has_action_custom_recommended_destinations'] = sessions_has_action(data_df, sessions_df, 'custom_recommended_destinations')
    data_df['s_has_action_pending'] = sessions_has_action(data_df, sessions_df, 'pending')
    data_df['s_has_action_profile_pic'] = sessions_has_action(data_df, sessions_df, 'profile_pic')
    data_df['s_has_action_populate_help_dropdown'] = sessions_has_action(data_df, sessions_df, 'populate_help_dropdown')
    data_df['s_has_action_message_to_host_change'] = sessions_has_action(data_df, sessions_df, 'message_to_host_change')
    data_df['s_has_action_ajax_image_upload'] = sessions_has_action(data_df, sessions_df, 'ajax_image_upload')
    data_df['s_has_action_view'] = sessions_has_action(data_df, sessions_df, 'view')
    data_df['s_has_action_kba_update'] = sessions_has_action(data_df, sessions_df, 'kba_update')
    data_df['s_has_action_references'] = sessions_has_action(data_df, sessions_df, 'references')
    data_df['s_has_action_my'] = sessions_has_action(data_df, sessions_df, 'my')
    data_df['s_has_action_ajax_get_referrals_amt'] = sessions_has_action(data_df, sessions_df, 'ajax_get_referrals_amt')
    data_df['s_has_action_new'] = sessions_has_action(data_df, sessions_df, 'new')
    data_df['s_has_action_agree_terms_check'] = sessions_has_action(data_df, sessions_df, 'agree_terms_check')
    data_df['s_has_action_apply_reservation'] = sessions_has_action(data_df, sessions_df, 'apply_reservation')
    data_df['s_has_action_connect'] = sessions_has_action(data_df, sessions_df, 'connect')
    data_df['s_has_action_recommended_listings'] = sessions_has_action(data_df, sessions_df, 'recommended_listings')
    data_df['s_has_action_faq'] = sessions_has_action(data_df, sessions_df, 'faq')
    data_df['s_has_action_populate_from_facebook'] = sessions_has_action(data_df, sessions_df, 'populate_from_facebook')
    data_df['s_has_action_account'] = sessions_has_action(data_df, sessions_df, 'account')
    data_df['s_has_action_available'] = sessions_has_action(data_df, sessions_df, 'available')
    data_df['s_has_action_jumio_token'] = sessions_has_action(data_df, sessions_df, 'jumio_token')
    data_df['s_has_action_qt_reply_v2'] = sessions_has_action(data_df, sessions_df, 'qt_reply_v2')
    data_df['s_has_action_signup_login'] = sessions_has_action(data_df, sessions_df, 'signup_login')
    data_df['s_has_action_request_new_confirm_email'] = sessions_has_action(data_df, sessions_df, 'request_new_confirm_email')
    data_df['s_has_action_kba'] = sessions_has_action(data_df, sessions_df, 'kba')
    data_df['s_has_action_handle_vanity_url'] = sessions_has_action(data_df, sessions_df, 'handle_vanity_url')
    data_df['s_has_action_coupon_field_focus'] = sessions_has_action(data_df, sessions_df, 'coupon_field_focus')
    data_df['s_has_action_phone_number_widget'] = sessions_has_action(data_df, sessions_df, 'phone_number_widget')
    data_df['s_has_action_open_graph_setting'] = sessions_has_action(data_df, sessions_df, 'open_graph_setting')
    data_df['s_has_action_set_user'] = sessions_has_action(data_df, sessions_df, 'set_user')
    data_df['s_has_action_faq_category'] = sessions_has_action(data_df, sessions_df, 'faq_category')

    data_df['s_has_view'] = sessions_has_action_type(data_df, sessions_df, 'view')
    data_df['s_has_data'] = sessions_has_action_type(data_df, sessions_df, 'data')
    data_df['s_has_click'] = sessions_has_action_type(data_df, sessions_df, 'click')
    data_df['s_has_unknown'] = sessions_has_action_type(data_df, sessions_df, '-unknown-')
    data_df['s_has_message_post'] = sessions_has_action_type(data_df, sessions_df, 'message_post')
    data_df['s_has_submit'] = sessions_has_action_type(data_df, sessions_df, 'submit')
    data_df['s_has_booking_request'] = sessions_has_action_type(data_df, sessions_df, 'booking_request')
    data_df['s_has_modify'] = sessions_has_action_type(data_df, sessions_df, 'modify')
    data_df['s_has_partner_callback'] = sessions_has_action_type(data_df, sessions_df, 'partner_callback')

    data_df['s_has_action_detail_view_search_results'] = sessions_has_action_detail(data_df, sessions_df, 'view_search_results')
    data_df['s_has_action_detail_p3'] = sessions_has_action_detail(data_df, sessions_df, 'p3')
    data_df['s_has_action_detail_unknown'] = sessions_has_action_detail(data_df, sessions_df, '-unknown-')
    data_df['s_has_action_detail_wishlist_content_update'] = sessions_has_action_detail(data_df, sessions_df, 'wishlist_content_update')
    data_df['s_has_action_detail_user_profile'] = sessions_has_action_detail(data_df, sessions_df, 'user_profile')
    data_df['s_has_action_detail_change_trip_characteristics'] = sessions_has_action_detail(data_df, sessions_df, 'change_trip_characteristics')
    data_df['s_has_action_detail_similar_listings'] = sessions_has_action_detail(data_df, sessions_df, 'similar_listings')
    data_df['s_has_action_detail_update_listing'] = sessions_has_action_detail(data_df, sessions_df, 'update_listing')
    data_df['s_has_action_detail_listing_reviews'] = sessions_has_action_detail(data_df, sessions_df, 'listing_reviews')
    data_df['s_has_action_detail_dashboard'] = sessions_has_action_detail(data_df, sessions_df, 'dashboard')
    data_df['s_has_action_detail_user_wishlists'] = sessions_has_action_detail(data_df, sessions_df, 'user_wishlists')
    data_df['s_has_action_detail_header_userpic'] = sessions_has_action_detail(data_df, sessions_df, 'header_userpic')
    data_df['s_has_action_detail_message_thread'] = sessions_has_action_detail(data_df, sessions_df, 'message_thread')
    data_df['s_has_action_detail_edit_profile'] = sessions_has_action_detail(data_df, sessions_df, 'edit_profile')
    data_df['s_has_action_detail_message_post'] = sessions_has_action_detail(data_df, sessions_df, 'message_post')
    data_df['s_has_action_detail_contact_host'] = sessions_has_action_detail(data_df, sessions_df, 'contact_host')
    data_df['s_has_action_detail_unavailable_dates'] = sessions_has_action_detail(data_df, sessions_df, 'unavailable_dates')
    data_df['s_has_action_detail_confirm_email_link'] = sessions_has_action_detail(data_df, sessions_df, 'confirm_email_link')
    data_df['s_has_action_detail_create_user'] = sessions_has_action_detail(data_df, sessions_df, 'create_user')
    data_df['s_has_action_detail_change_contact_host_dates'] = sessions_has_action_detail(data_df, sessions_df, 'change_contact_host_dates')
    data_df['s_has_action_detail_user_profile_content_update'] = sessions_has_action_detail(data_df, sessions_df, 'user_profile_content_update')
    data_df['s_has_action_detail_user_reviews'] = sessions_has_action_detail(data_df, sessions_df, 'user_reviews')
    data_df['s_has_action_detail_p5'] = sessions_has_action_detail(data_df, sessions_df, 'p5')
    data_df['s_has_action_detail_login'] = sessions_has_action_detail(data_df, sessions_df, 'login')
    data_df['s_has_action_detail_your_trips'] = sessions_has_action_detail(data_df, sessions_df, 'your_trips')
    data_df['s_has_action_detail_p1'] = sessions_has_action_detail(data_df, sessions_df, 'p1')

    print('add_sessions_data >>')

    return data_df




def print_columns(columns):
    print('columns: ', ['f'+str(idx)+': '+col for idx, col in enumerate(columns)])